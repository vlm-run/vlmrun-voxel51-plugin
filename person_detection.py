"""Person detection operator for VLM Run Plugin."""

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


class VLMRunPersonDetection(foo.Operator):
    """Detect persons in images using VLM Run's person detection."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="vlmrun_person_detection",
            label="VLM Run: Person Detection",
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

        # Fixed field name for person detection
        default_field = "person_detections"

        inputs.str(
            "result_field",
            label="Result Field",
            description="Field name to store detected persons",
            default=default_field,
            required=True,
        )

        return types.Property(
            inputs, view=types.View(label="Person Detection")
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
        domain = "image.person-detection"  # Fixed domain

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
                "error": "VLMRun package not installed. Run: fiftyone plugins requirements @voxel51/vlmrun --install"
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
                    self._process_person_result(
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

    def _process_person_result(self, sample, result, result_field):
        """Process VLM Run person detection result and update sample."""

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
            # Store the content description
            if "content" in response_data:
                sample[f"{result_field}_description"] = response_data["content"]

            detections = []

            # Person detection returns fields like "person-1_page0_metadata" with bboxes
            for key, value in response_data.items():
                if key.endswith("_metadata") and "person" in key:
                    if isinstance(value, dict) and "bboxes" in value:
                        # Extract confidence from metadata
                        confidence_str = value.get("confidence", "med")
                        if confidence_str == "hi":
                            confidence = 0.9
                        elif confidence_str == "med":
                            confidence = 0.7
                        else:
                            confidence = 0.5

                        for bbox_info in value["bboxes"]:
                            if isinstance(bbox_info, dict):
                                # Extract bbox coordinates
                                bbox_data = None
                                if "bbox" in bbox_info and "xywh" in bbox_info["bbox"]:
                                    bbox_data = bbox_info["bbox"]["xywh"]
                                elif "xywh" in bbox_info:
                                    bbox_data = bbox_info["xywh"]

                                if bbox_data:
                                    detection = fol.Detection(
                                        label=bbox_info.get("content", "person"),
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
                default=f"Successfully detected persons in {ctx.results.get('processed')} image(s). Check the '{ctx.params.get('result_field', 'person_detections')}' field in your samples.",
                view=types.Notice(variant="success"),
            )

        return types.Property(
            outputs, view=types.View(label="Person Detection Results")
        )
