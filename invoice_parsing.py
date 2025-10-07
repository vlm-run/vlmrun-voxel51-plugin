"""Invoice parsing operator for VLM Run Plugin."""

import os
import os.path
import time
from pathlib import Path

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.utils as fou
import fiftyone.core.labels as fol
from vlmrun.client.types import GenerationConfig

# Configuration constants
DEFAULT_API_URL = "https://api.vlm.run/v1"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_WAIT = 600  # 10 minutes
DEFAULT_POLL_INTERVAL = 5  # seconds
MAX_ERROR_DETAILS = 5

# Supported file extensions
DOCUMENT_EXTENSIONS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
)


class VLMRunParseInvoices(foo.Operator):
    """Extract structured data from invoices using VLM Run."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="vlmrun_parse_invoices",
            label="VLM Run: Parse Invoices",
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

        # Fixed field name for invoice parsing
        default_field = "invoice_data"

        inputs.str(
            "result_field",
            label="Result Field",
            description="Field name to store invoice data",
            default=default_field,
            required=True,
        )

        # Grounding option
        inputs.bool(
            "enable_grounding",
            label="Enable Visual Grounding",
            description="Extract bounding boxes for detected invoice fields",
            default=True,
            required=False,
        )

        inputs.str(
            "detections_field",
            label="Detections Field",
            description="Field name to store bounding box detections (if grounding enabled)",
            default="invoice_detections",
            required=False,
        )

        return types.Property(
            inputs, view=types.View(label="Parse Invoices")
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
        enable_grounding = ctx.params.get("enable_grounding", True)
        detections_field = ctx.params.get("detections_field", "invoice_detections")
        domain = "document.invoice"  # Fixed domain for this operator

        # Get samples
        sample_collection = ctx.view if target == "VIEW" else ctx.dataset

        # Filter for document samples
        document_samples = sample_collection
        total_documents = len(document_samples)

        if total_documents == 0:
            return {
                "error": "No document samples found in the selected collection"
            }

        # Initialize VLM Run client
        try:
            from vlmrun.client import VLMRun
        except ImportError:
            return {
                "error": "VLMRun package not installed. Run: fiftyone plugins requirements @vlm-run/vlmrun-voxel51-plugin --install"
            }

        # Get configuration from environment or use defaults
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

        # Create config for grounding if enabled
        config = None
        if enable_grounding:
            config = GenerationConfig(grounding=True)

        processed = 0
        errors = []

        with fou.ProgressBar(total=total_documents) as pb:
            for sample in document_samples:
                try:
                    # Skip non-document files
                    if not sample.filepath.lower().endswith(DOCUMENT_EXTENSIONS):
                        pb.update()
                        continue

                    # Process document with VLM Run
                    file_path = Path(sample.filepath)

                    # Use batch mode for documents as they may take longer
                    generate_kwargs = {
                        "file": file_path,
                        "domain": domain,
                        "batch": True,
                    }
                    if enable_grounding and config:
                        generate_kwargs["config"] = config

                    response = client.document.generate(**generate_kwargs)

                    # Poll for batch completion
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
                                    f"Document prediction failed: {pred_response.error if hasattr(pred_response, 'error') else 'Unknown error'}"
                                )

                            time.sleep(poll_interval)
                            elapsed += poll_interval
                        else:
                            raise TimeoutError(
                                f"Document prediction timed out after {max_wait} seconds"
                            )

                    else:
                        result = response

                    # Parse and store the result
                    self._process_invoice_result(
                        sample,
                        result,
                        result_field,
                        enable_grounding,
                        detections_field,
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

    def _process_invoice_result(self, sample, result, result_field, enable_grounding=False, detections_field=None):
        """Process VLM Run invoice result and update sample."""

        # Extract response data - handle nested response structure
        if hasattr(result, "response"):
            response_data = result.response
            # If response is a Pydantic model, convert to dict
            if hasattr(response_data, "model_dump"):
                response_data = response_data.model_dump()
        elif hasattr(result, "data"):
            response_data = result.data
        else:
            response_data = result

        # Store invoice results
        if isinstance(response_data, dict):

            # Collect detections if grounding is enabled
            detections = []

            # Store key invoice fields based on actual response
            if "invoice_id" in response_data:
                sample[f"{result_field}_id"] = response_data["invoice_id"]
                # Check for grounding metadata
                if enable_grounding and "invoice_id_metadata" in response_data:
                    self._add_detection_from_metadata(
                        detections, "invoice_id", response_data["invoice_id_metadata"]
                    )

            if "issuer" in response_data:
                sample[f"{result_field}_issuer"] = response_data["issuer"]
                if enable_grounding and "issuer_metadata" in response_data:
                    self._add_detection_from_metadata(
                        detections, "issuer", response_data["issuer_metadata"]
                    )

            if "customer" in response_data:
                sample[f"{result_field}_customer"] = response_data["customer"]
                if enable_grounding and "customer_metadata" in response_data:
                    self._add_detection_from_metadata(
                        detections, "customer", response_data["customer_metadata"]
                    )

            if "invoice_issue_date" in response_data:
                sample[f"{result_field}_date"] = response_data["invoice_issue_date"]
                if enable_grounding and "invoice_issue_date_metadata" in response_data:
                    self._add_detection_from_metadata(
                        detections, "invoice_date", response_data["invoice_issue_date_metadata"]
                    )

            if "total" in response_data:
                sample[f"{result_field}_total"] = response_data["total"]
                if enable_grounding and "total_metadata" in response_data:
                    self._add_detection_from_metadata(
                        detections, "total", response_data["total_metadata"]
                    )

            if "currency" in response_data:
                sample[f"{result_field}_currency"] = response_data["currency"]
                if enable_grounding and "currency_metadata" in response_data:
                    self._add_detection_from_metadata(
                        detections, "currency", response_data["currency_metadata"]
                    )

            if "items" in response_data:
                sample[f"{result_field}_items"] = response_data["items"]
                # Items might have their own metadata

            # Store detections if grounding is enabled and we have detections
            if enable_grounding and detections and detections_field:
                sample[detections_field] = fol.Detections(detections=detections)

            # Store full response for reference
            sample[result_field] = response_data
        else:
            sample[result_field] = str(response_data)

    def _add_detection_from_metadata(self, detections_list, label, metadata):
        """Convert VLM Run grounding metadata to FiftyOne Detection."""
        if not metadata or not isinstance(metadata, dict):
            return

        if "bboxes" in metadata:
            # Convert confidence string to numeric
            confidence_str = metadata.get("confidence", "med")
            if confidence_str == "hi":
                confidence = 0.9
            elif confidence_str == "med":
                confidence = 0.7
            else:  # "low"
                confidence = 0.5

            # Process each bounding box
            for bbox_info in metadata["bboxes"]:
                # Check for bbox in nested structure
                bbox_data = None
                if "bbox" in bbox_info and "xywh" in bbox_info["bbox"]:
                    bbox_data = bbox_info["bbox"]["xywh"]
                elif "xywh" in bbox_info:
                    bbox_data = bbox_info["xywh"]

                if bbox_data:
                    # VLM Run format is already [x, y, w, h] normalized
                    detection = fol.Detection(
                        label=label,
                        bounding_box=bbox_data,
                        confidence=confidence,
                    )
                    # Add page info if available
                    if "page" in bbox_info:
                        detection["page"] = bbox_info["page"]

                    detections_list.append(detection)

    def resolve_output(self, ctx):
        """Display output to the user."""
        outputs = types.Object()

        # Show actual results
        if "processed" in ctx.results:
            outputs.int("processed", label="Invoices Processed")
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
                default=f"Successfully parsed {ctx.results.get('processed')} invoice(s). Check the '{ctx.params.get('result_field', 'invoice_data')}' field in your samples.",
                view=types.Notice(variant="success"),
            )

        return types.Property(
            outputs, view=types.View(label="Invoice Parsing Results")
        )