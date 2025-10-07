"""Layout detection operator for VLM Run Plugin."""

import os
import os.path
import time
from pathlib import Path

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.utils as fou
import fiftyone.core.labels as fol

# Configuration constants
DEFAULT_API_URL = "https://api.vlm.run/v1"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_WAIT = 600  # 10 minutes
DEFAULT_POLL_INTERVAL = 5  # seconds
MAX_ERROR_DETAILS = 5


class VLMRunLayoutDetection(foo.Operator):
    """Detect document layout elements using VLM Run."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="vlmrun_layout_detection",
            label="VLM Run: Document Layout Detection",
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=False,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # Check for API key
        api_key = ctx.secrets.get(
            "VLMRUN_API_KEY", os.getenv("VLMRUN_API_KEY")
        )
        if not api_key:
            inputs.str(
                "api_key",
                label="VLM Run API Key",
                description="Your VLM Run API key",
                required=True,
            )

        # Target selection
        has_view = ctx.dataset is not None and ctx.view != ctx.dataset.view()
        if has_view:
            target_choices = types.RadioGroup()
            target_choices.add_choice("DATASET", label="Entire dataset")
            target_choices.add_choice("VIEW", label="Current view")
            inputs.enum(
                "target",
                target_choices.values(),
                default="VIEW",
                label="Process",
                view=target_choices,
            )

        # Fixed field name for layout detection
        default_field = "layout_detections"

        inputs.str(
            "result_field",
            label="Result Field",
            description="Field name to store detected layout elements",
            default=default_field,
            required=True,
        )

        return types.Property(
            inputs, view=types.View(label="Layout Detection")
        )

    def execute(self, ctx):
        # Get parameters
        api_key = ctx.params.get("api_key") or ctx.secrets.get(
            "VLMRUN_API_KEY", os.getenv("VLMRUN_API_KEY")
        )

        if not api_key:
            return {"error": "VLM Run API key is required"}

        target = ctx.params.get("target", "DATASET")
        result_field = ctx.params["result_field"]
        domain = "document.layout-detection"  # Fixed domain

        # Get samples
        sample_collection = ctx.view if target == "VIEW" else ctx.dataset
        document_samples = sample_collection
        total_documents = len(document_samples)

        if total_documents == 0:
            return {
                "error": "No document samples found in the selected collection"
            }

        # Initialize VLM Run client
        try:
            from vlmrun.client import VLMRun
            from vlmrun.client.types import GenerationConfig
        except ImportError:
            return {
                "error": "VLMRun package not installed. Run: fiftyone plugins requirements @vlm-run/vlmrun-voxel51-plugin --install"
            }

        # Get configuration
        api_url = os.getenv("VLMRUN_API_URL", DEFAULT_API_URL)
        timeout = float(os.getenv("VLMRUN_TIMEOUT", str(DEFAULT_TIMEOUT)))
        max_retries = int(
            os.getenv("VLMRUN_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))
        )

        client = VLMRun(
            api_key=api_key,
            base_url=api_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        processed = 0
        errors = []

        # Create config with grounding enabled
        config = GenerationConfig(grounding=True)

        with fou.ProgressBar(total=total_documents) as pb:
            for sample in document_samples:
                try:
                    # Process document with VLM Run
                    file_path = Path(sample.filepath)

                    # Use batch mode for documents
                    response = client.document.generate(
                        file=file_path,
                        domain=domain,
                        config=config,
                        batch=True,
                    )

                    # Poll for batch completion if needed
                    if hasattr(response, "id") and hasattr(response, "status"):
                        prediction_id = response.id
                        max_wait = int(
                            os.getenv("VLMRUN_MAX_WAIT", str(DEFAULT_MAX_WAIT))
                        )
                        poll_interval = int(
                            os.getenv(
                                "VLMRUN_POLL_INTERVAL",
                                str(DEFAULT_POLL_INTERVAL),
                            )
                        )
                        elapsed = 0

                        while elapsed < max_wait:
                            pred_response = client.predictions.get(
                                id=prediction_id
                            )

                            if pred_response.status == "completed":
                                result = (
                                    pred_response.result
                                    if hasattr(pred_response, "result")
                                    else pred_response
                                )
                                break
                            elif pred_response.status == "failed":
                                raise RuntimeError(
                                    f"Layout detection failed: {pred_response.error if hasattr(pred_response, 'error') else 'Unknown error'}"
                                )

                            time.sleep(poll_interval)
                            elapsed += poll_interval
                        else:
                            raise TimeoutError(
                                f"Layout detection timed out after {max_wait} seconds"
                            )
                    else:
                        result = response

                    # Parse and store the result
                    self._process_layout_result(
                        sample,
                        result,
                        result_field,
                    )

                    sample.save()
                    processed += 1

                except Exception as e:
                    error_msg = f"Failed to process {os.path.basename(sample.filepath)}: {str(e)}"
                    errors.append(error_msg)

                pb.update()

        # Refresh the app
        if not ctx.delegated:
            ctx.trigger("reload_dataset")

        # Return summary
        result = {
            "processed": processed,
            "total": total_documents,
            "errors": len(errors),
        }

        if errors:
            result["error_details"] = errors[:MAX_ERROR_DETAILS]

        return result

    def _process_layout_result(self, sample, result, result_field):
        """Process VLM Run layout detection result and update sample."""

        # Extract response data
        if hasattr(result, "response"):
            response_data = result.response
            if hasattr(response_data, "model_dump"):
                response_data = response_data.model_dump()
        elif hasattr(result, "data"):
            response_data = result.data
        else:
            response_data = result

        if isinstance(response_data, dict):
            detections = []
            layout_elements = {}

            # Process layout elements - they come as key-value pairs with metadata
            for key, value in response_data.items():
                if not key.endswith("_metadata"):
                    # Store the layout element text
                    element_name = key.replace("_page0", "")
                    layout_elements[element_name] = value

                    # Check for corresponding metadata
                    metadata_key = f"{key}_metadata"
                    if metadata_key in response_data:
                        metadata = response_data[metadata_key]
                        if isinstance(metadata, dict) and "bboxes" in metadata:
                            for bbox_info in metadata["bboxes"]:
                                if "bbox" in bbox_info and "xywh" in bbox_info["bbox"]:
                                    bbox_data = bbox_info["bbox"]["xywh"]

                                    # Convert confidence to numeric
                                    confidence_str = metadata.get("confidence", "med")
                                    if confidence_str == "hi":
                                        confidence = 0.9
                                    elif confidence_str == "med":
                                        confidence = 0.7
                                    else:
                                        confidence = 0.5

                                    detection = fol.Detection(
                                        label=element_name,
                                        bounding_box=bbox_data,
                                        confidence=confidence,
                                    )

                                    # Add page info if available
                                    if "page" in bbox_info:
                                        detection["page"] = bbox_info["page"]

                                    detections.append(detection)

            # Store layout elements as structured data
            if layout_elements:
                sample[f"{result_field}_elements"] = layout_elements

            # Store detections
            if detections:
                sample[result_field] = fol.Detections(detections=detections)

    def resolve_output(self, ctx):
        """Display output to the user."""
        outputs = types.Object()

        # Show actual results
        if "processed" in ctx.results:
            outputs.int("processed", label="Documents Processed")
        if "total" in ctx.results:
            outputs.int("total", label="Total Documents")
        if "errors" in ctx.results:
            outputs.int("errors", label="Errors")
        if "error" in ctx.results:
            outputs.str("error", label="Error", view=types.Warning())
        if "error_details" in ctx.results:
            outputs.list(
                "error_details", types.String(), label="Error Details"
            )

        # Success message
        if ctx.results.get("processed", 0) > 0:
            outputs.str(
                "success_msg",
                label="Success",
                default=f"Successfully detected layout in {ctx.results.get('processed')} document(s). Check the '{ctx.params.get('result_field', 'layout_detections')}' field in your samples.",
                view=types.Notice(variant="success"),
            )

        return types.Property(
            outputs, view=types.View(label="Layout Detection Results")
        )