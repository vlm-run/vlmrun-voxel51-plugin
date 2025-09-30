"""Image captioning operator for VLM Run Plugin."""

import os
import os.path
from pathlib import Path
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.utils as fou

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


class VLMRunCaptionImages(foo.Operator):
    """Generate descriptive captions for images using VLM Run."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="vlmrun_caption_images",
            label="VLM Run: Caption Images",
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

        # Fixed field name for captions
        default_field = "image_caption"

        inputs.str(
            "result_field",
            label="Result Field",
            description="Field name to store image captions",
            default=default_field,
            required=True,
        )

        inputs.bool(
            "populate_builtin_tags",
            label="Populate Built-in Tags",
            description="If the caption includes tags, also add them to the sample's built-in 'tags' field for easier filtering",
            default=False,
            required=False,
        )

        return types.Property(
            inputs, view=types.View(label="Caption Images")
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
        populate_builtin_tags = ctx.params.get("populate_builtin_tags", False)
        domain = "image.caption"  # Fixed domain for this operator

        # Get samples
        sample_collection = ctx.view if target == "VIEW" else ctx.dataset

        # Filter for image samples
        image_samples = sample_collection
        total_images = len(image_samples)

        if total_images == 0:
            return {
                "error": "No image samples found in the selected collection"
            }

        # Initialize VLM Run client
        try:
            from vlmrun.client import VLMRun
        except ImportError:
            return {
                "error": "VLMRun package not installed. Run: fiftyone plugins requirements @voxel51/vlmrun --install"
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

        processed = 0
        errors = []

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
                    )

                    # Parse and store the result
                    self._process_image_result(
                        sample,
                        response,
                        result_field,
                        domain,
                        populate_builtin_tags=populate_builtin_tags,
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

    def _process_image_result(self, sample, result, result_field, domain, populate_builtin_tags=False):
        """Process VLM Run image result and update sample."""

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

        # Store caption results
        if isinstance(response_data, dict):
            # Store caption as main result
            if "caption" in response_data:
                sample[result_field] = response_data["caption"]
            elif "description" in response_data:
                sample[result_field] = response_data["description"]

            # Store tags if present
            if "tags" in response_data:
                sample[f"{result_field}_tags"] = response_data["tags"]

                # Optionally populate the built-in tags field
                if populate_builtin_tags:
                    if hasattr(sample, 'tags'):
                        # Append to existing tags
                        existing_tags = sample.tags if sample.tags else []
                        sample.tags = list(set(existing_tags + response_data["tags"]))
                    else:
                        sample.tags = response_data["tags"]
        else:
            sample[result_field] = str(response_data)

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
                default=f"Successfully captioned {ctx.results.get('processed')} image(s). Check the '{ctx.params.get('result_field', 'image_caption')}' field in your samples.",
                view=types.Notice(variant="success"),
            )

        return types.Property(
            outputs, view=types.View(label="Captioning Results")
        )