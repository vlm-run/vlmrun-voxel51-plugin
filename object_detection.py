"""Object detection operator for VLM Run Plugin."""

import os
import os.path
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
MAX_ERROR_DETAILS = 5

# Supported file extensions
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
    ".webp",
)


class VLMRunObjectDetection(foo.Operator):
    """Detect objects in images using VLM Run's object detection."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="vlmrun_object_detection",
            label="VLM Run: Object Detection",
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

        # Fixed field name for object detection
        default_field = "object_detections"

        inputs.str(
            "result_field",
            label="Result Field",
            description="Field name to store detected objects",
            default=default_field,
            required=True,
        )

        return types.Property(
            inputs, view=types.View(label="Object Detection")
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
        domain = "image.object-detection"  # Fixed domain

        # Get samples
        sample_collection = ctx.view if target == "VIEW" else ctx.dataset
        image_samples = sample_collection
        total_images = len(image_samples)

        if total_images == 0:
            return {
                "error": "No image samples found in the selected collection"
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

        with fou.ProgressBar(total=total_images) as pb:
            for sample in image_samples:
                try:
                    # Skip non-image files
                    if not sample.filepath.lower().endswith(IMAGE_EXTENSIONS):
                        pb.update()
                        continue

                    # Process image with VLM Run
                    file_path = Path(sample.filepath)

                    response = client.image.generate(
                        images=[file_path],
                        domain=domain,
                        config=config,
                    )

                    # Parse and store the result
                    self._process_detection_result(
                        sample,
                        response,
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
            "total": total_images,
            "errors": len(errors),
        }

        if errors:
            result["error_details"] = errors[:MAX_ERROR_DETAILS]

        return result

    def _process_detection_result(self, sample, result, result_field):
        """Process VLM Run object detection result and update sample."""

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

            # Store the content description
            if "content" in response_data:
                sample[f"{result_field}_description"] = response_data["content"]

            # Process detected objects - they come as metadata fields
            for key, value in response_data.items():
                if key.endswith("_metadata") and isinstance(value, dict):
                    # Extract object label from the key
                    label = key.replace("_metadata", "").replace("_page0", "")

                    if "bboxes" in value:
                        for bbox_info in value["bboxes"]:
                            if "bbox" in bbox_info and "xywh" in bbox_info["bbox"]:
                                bbox_data = bbox_info["bbox"]["xywh"]

                                # Convert confidence to numeric
                                confidence_str = value.get("confidence", "med")
                                if confidence_str == "hi":
                                    confidence = 0.9
                                elif confidence_str == "med":
                                    confidence = 0.7
                                else:
                                    confidence = 0.5

                                detection = fol.Detection(
                                    label=label,
                                    bounding_box=bbox_data,
                                    confidence=confidence,
                                )
                                detections.append(detection)

            if detections:
                sample[result_field] = fol.Detections(detections=detections)

    def resolve_output(self, ctx):
        """Display output to the user."""
        outputs = types.Object()

        # Show actual results
        if "processed" in ctx.results:
            outputs.int("processed", label="Images Processed")
        if "total" in ctx.results:
            outputs.int("total", label="Total Images")
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
                default=f"Successfully detected objects in {ctx.results.get('processed')} image(s). Check the '{ctx.params.get('result_field', 'object_detections')}' field in your samples.",
                view=types.Notice(variant="success"),
            )

        return types.Property(
            outputs, view=types.View(label="Object Detection Results")
        )