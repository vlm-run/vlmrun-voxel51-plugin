"""Chat completions operator for VLM Run Plugin using Orion API.

This module provides a FiftyOne operator for analyzing media using VLM Run's
Orion chat completions API, supporting natural language prompts and multiple
response formats including text, JSON, and schema-validated JSON.

Supported formats:
- Images (JPG, PNG, WebP)
- Videos (MP4)
- Documents (PDF)
- Audio: NOT supported in chat completions (use client.audio instead)

All supported files are uploaded and referenced by file_id using the input_file type.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.utils as fou

# Configuration constants
DEFAULT_AGENT_API_URL = "https://agent.vlm.run/v1"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 5
MAX_ERROR_DETAILS = 5

# Default model
DEFAULT_MODEL = "vlmrun-orion-1:auto"

# Available Orion models
ORION_MODELS = [
    ("vlmrun-orion-1:fast", "Fast - Optimized for simple tasks with speed"),
    ("vlmrun-orion-1:auto", "Auto - Automatically selects best model for task"),
    ("vlmrun-orion-1:pro", "Pro - Most capable for complex multi-step workflows"),
]

# Supported file extensions (per VLM Run docs)
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
)

VIDEO_EXTENSIONS = (
    ".mp4",
)

DOCUMENT_EXTENSIONS = (
    ".pdf",
)

AUDIO_EXTENSIONS = (
    ".mp3",
)

# Note: Audio is NOT supported in chat completions
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS + DOCUMENT_EXTENSIONS


class VLMRunChatCompletions(foo.Operator):
    """Analyze media using VLM Run's Orion chat completions API.

    This operator uses VLM Run's OpenAI-compatible chat completions endpoint
    to perform flexible analysis with natural language prompts.

    Supported formats:
    - Images (JPG, PNG, WebP)
    - Videos (MP4)
    - Documents (PDF)
    - Audio: NOT supported (use standard VLM Run audio endpoints instead)

    Files are uploaded and referenced by file_id using the input_file type.
    Supports text, JSON object, and JSON schema response formats.
    """

    @property
    def config(self) -> foo.OperatorConfig:
        """Return operator configuration.

        Returns:
            OperatorConfig with operator name, label, and settings.
        """
        return foo.OperatorConfig(
            name="vlmrun_chat_completions",
            label="VLM Run: Chat Completions (Orion)",
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx: foo.ExecutionContext) -> types.Property:
        """Define the input form schema for the operator.

        Args:
            ctx: The execution context containing dataset and view info.

        Returns:
            Property defining the input form.
        """
        inputs = types.Object()

        # Check for API key
        api_key = ctx.secrets.get(
            "VLMRUN_API_KEY", os.getenv("VLMRUN_API_KEY")
        )
        if not api_key:
            inputs.str(
                "api_key",
                label="VLM Run API Key",
                description="Your VLM Run API key (get one at https://vlm.run)",
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

        # Model selection
        model_choices = types.Dropdown()
        for model_id, model_desc in ORION_MODELS:
            model_choices.add_choice(model_id, label=model_desc)

        inputs.enum(
            "model",
            model_choices.values(),
            default=DEFAULT_MODEL,
            label="Model",
            description="Select the Orion model variant to use",
            view=model_choices,
        )

        # Prompt input
        inputs.str(
            "prompt",
            label="Prompt",
            description="Natural language prompt for media analysis (e.g., 'Describe what you see')",
            default="Describe what you see in detail.",
            required=True,
        )

        # Response format selection
        response_format_choices = types.Dropdown()
        response_format_choices.add_choice(
            "text", label="Text - Free-form text response"
        )
        response_format_choices.add_choice(
            "json_object", label="JSON Object - Structured JSON output"
        )

        inputs.enum(
            "response_format",
            response_format_choices.values(),
            default="text",
            label="Response Format",
            description="Choose how the model should format its response",
            view=response_format_choices,
        )

        # Result field name
        default_field = "chat_response"
        inputs.str(
            "result_field",
            label="Result Field",
            description="Field name to store the chat completion response",
            default=default_field,
            required=True,
        )

        # Temperature
        inputs.float(
            "temperature",
            label="Temperature",
            description="Controls randomness (0 = deterministic, 1 = creative). Use 0 for structured outputs.",
            default=0.0,
            required=False,
        )

        # System prompt (optional)
        inputs.str(
            "system_prompt",
            label="System Prompt (Optional)",
            description="Optional system prompt to set context for the model",
            required=False,
        )

        # Max samples (for testing/limiting)
        inputs.int(
            "max_samples",
            label="Max Samples (Optional)",
            description="Limit number of samples to process (leave empty for all)",
            required=False,
        )

        # Option to save generated images as new samples
        inputs.bool(
            "save_generated_images",
            label="Save Generated Images",
            description="If enabled, any images generated by the model (e.g., extracted frames, visualizations) will be saved and added as new samples to the dataset.",
            default=False,
        )

        return types.Property(
            inputs, view=types.View(label="Chat Completions (Orion)")
        )

    def execute(self, ctx: foo.ExecutionContext) -> Dict[str, Any]:
        """Execute the chat completions operator.

        Args:
            ctx: The execution context with parameters and dataset.

        Returns:
            Dictionary with processing results including counts and errors.
        """
        # Get parameters
        api_key = ctx.params.get("api_key") or ctx.secrets.get(
            "VLMRUN_API_KEY", os.getenv("VLMRUN_API_KEY")
        )

        if not api_key:
            return {"error": "VLM Run API key is required"}

        target = ctx.params.get("target", "DATASET")
        model = ctx.params.get("model", DEFAULT_MODEL)
        prompt = ctx.params["prompt"]
        response_format = ctx.params.get("response_format", "text")
        result_field = ctx.params["result_field"]
        temperature = ctx.params.get("temperature", 0.0)
        system_prompt = ctx.params.get("system_prompt")
        max_samples = ctx.params.get("max_samples")
        save_generated_images = ctx.params.get("save_generated_images", False)

        # Auto-determine output directory for generated images
        output_dir = None
        if save_generated_images:
            # Try to find a sensible default based on dataset samples
            first_sample = ctx.dataset.first()
            if first_sample and first_sample.filepath:
                # Create a 'generated' subdirectory next to the source samples
                source_dir = Path(first_sample.filepath).parent
                output_dir = source_dir / "generated_outputs"
            else:
                # Fallback to a temp directory with dataset name
                import tempfile
                output_dir = Path(tempfile.gettempdir()) / f"fiftyone_generated_{ctx.dataset.name}"

            output_dir.mkdir(parents=True, exist_ok=True)

        # Get samples
        sample_collection = ctx.view if target == "VIEW" else ctx.dataset

        # Apply max_samples limit if specified
        if max_samples and max_samples > 0:
            samples = sample_collection.take(max_samples)
        else:
            samples = sample_collection

        total_samples = len(samples)

        if total_samples == 0:
            return {
                "error": "No supported samples found in the selected collection (images, videos, or PDFs)"
            }

        # Initialize VLM Run client with Orion API endpoint
        try:
            from vlmrun.client import VLMRun
        except ImportError:
            return {
                "error": "VLMRun package not installed. Run: fiftyone plugins requirements @vlm-run/vlmrun-voxel51-plugin --install"
            }

        # Get configuration
        api_url = os.getenv("VLMRUN_AGENT_API_URL", DEFAULT_AGENT_API_URL)
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
        generated_samples = 0
        errors = []

        with fou.ProgressBar(total=total_samples) as pb:
            for sample in samples:
                try:
                    # Skip unsupported file types
                    if not sample.filepath.lower().endswith(SUPPORTED_EXTENSIONS):
                        pb.update()
                        continue

                    # Build messages
                    messages = []

                    # Add system prompt if provided
                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt
                        })

                    # Build user message with media
                    file_path = Path(sample.filepath)

                    # Get media content with correct type for the API
                    media_content = self._get_media_content(file_path, client)

                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            media_content
                        ]
                    })

                    # Build request kwargs
                    request_kwargs: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                    }

                    # Add response format if not text
                    if response_format == "json_object":
                        request_kwargs["response_format"] = {"type": "json_object"}

                    # Make the API call using Orion agent completions
                    response = client.agent.completions.create(**request_kwargs)

                    # Extract and store the result
                    self._process_chat_result(
                        sample,
                        response,
                        result_field,
                        response_format,
                    )

                    sample.save()
                    processed += 1

                    # Extract output images and add as new samples if output_dir is provided
                    if output_dir:
                        saved_images = self._extract_output_images(
                            response, client, sample, output_dir
                        )
                        for image_path in saved_images:
                            # Create new sample for the generated image
                            new_sample = fo.Sample(filepath=image_path)
                            new_sample.tags.append("vlmrun_generated")
                            new_sample["source_sample_id"] = str(sample.id)
                            new_sample["source_filepath"] = sample.filepath
                            new_sample["generated_by"] = "vlmrun_chat_completions"
                            new_sample["prompt"] = prompt
                            ctx.dataset.add_sample(new_sample)
                            generated_samples += 1

                except Exception as e:
                    error_msg = f"Failed to process {os.path.basename(sample.filepath)}: {str(e)}"
                    errors.append(error_msg)

                pb.update()

        # Refresh the app
        if not ctx.delegated:
            ctx.trigger("reload_dataset")

        # Return summary
        result: Dict[str, Any] = {
            "processed": processed,
            "total": total_samples,
            "errors": len(errors),
        }

        if generated_samples > 0:
            result["generated_samples"] = generated_samples
            result["output_directory"] = str(output_dir)

        if errors:
            result["error_details"] = errors[:MAX_ERROR_DETAILS]

        return result

    def _get_media_content(self, file_path: Path, client: Any) -> Dict[str, Any]:
        """Build media content object with correct type for the Orion API.

        Uploads the file and uses input_file type with file_id for all media types.
        This is cleaner than using base64/URLs and works for images, videos, and PDFs.

        Args:
            file_path: Path to the media file.
            client: VLMRun client for uploading files.

        Returns:
            Dictionary with input_file type and file_id for the API.
        """
        # Upload file and use input_file type with file_id
        uploaded_file = client.files.upload(file=file_path)
        return {
            "type": "input_file",
            "file_id": uploaded_file.id
        }

    def _extract_output_images(
        self,
        response: Any,
        client: Any,
        source_sample: fo.Sample,
        output_dir: Path,
    ) -> list:
        """Extract and save any output images from the response.

        Detects image artifact references (img_XXXXXX) in the response content,
        downloads them using the artifacts API, and saves them to disk.

        Args:
            response: The API response containing potential image refs.
            client: VLMRun client for downloading artifacts.
            source_sample: The original sample that generated this output.
            output_dir: Directory to save output images.

        Returns:
            List of file paths to saved images.
        """
        import re

        saved_images = []

        # Get response content
        if not hasattr(response, "choices") or not response.choices:
            return saved_images

        content = response.choices[0].message.content or ""
        session_id = getattr(response, "session_id", None)

        if not session_id:
            return saved_images

        # Find image artifact references (pattern: img_XXXXXX)
        image_refs = re.findall(r'img_[a-zA-Z0-9]{6}', content)

        if not image_refs:
            return saved_images

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_ref in image_refs:
            try:
                # Download the image artifact
                image = client.artifacts.get(
                    session_id=session_id,
                    object_id=img_ref,
                )

                # Generate filename based on source sample and artifact ID
                source_name = Path(source_sample.filepath).stem
                output_path = output_dir / f"{source_name}_{img_ref}.png"

                # Save the image
                image.save(str(output_path))
                saved_images.append(str(output_path))

            except Exception as e:
                # Log but don't fail on artifact download errors
                pass

        return saved_images

    def _process_chat_result(
        self,
        sample: fo.Sample,
        result: Any,
        result_field: str,
        response_format: str
    ) -> None:
        """Process VLM Run chat completion result and update sample.

        Args:
            sample: The FiftyOne sample to update.
            result: The API response from VLM Run.
            result_field: The field name to store results.
            response_format: The response format (text or json_object).
        """
        # Extract the response content
        if hasattr(result, "choices") and result.choices:
            content = result.choices[0].message.content
        else:
            content = str(result)

        # Parse JSON responses
        if response_format == "json_object":
            try:
                parsed_content = json.loads(content)
                sample[result_field] = parsed_content
            except json.JSONDecodeError:
                # Store as string if JSON parsing fails
                sample[result_field] = content
        else:
            # Store text response directly
            sample[result_field] = content

        # Store model info as metadata
        if hasattr(result, "model"):
            sample[f"{result_field}_model"] = result.model

        # Store usage info if available
        if hasattr(result, "usage") and result.usage:
            usage_data = {
                token_type: getattr(result.usage, token_type)
                for token_type in ("prompt_tokens", "completion_tokens", "total_tokens")
                if hasattr(result.usage, token_type)
            }
            if usage_data:
                sample[f"{result_field}_usage"] = usage_data

    def resolve_output(self, ctx: foo.ExecutionContext) -> types.Property:
        """Define the output display schema.

        Args:
            ctx: The execution context with results.

        Returns:
            Property defining the output display.
        """
        outputs = types.Object()

        # Show actual results
        if "processed" in ctx.results:
            outputs.int("processed", label="Samples Processed")
        if "total" in ctx.results:
            outputs.int("total", label="Total Samples")
        if "generated_samples" in ctx.results:
            outputs.int("generated_samples", label="Generated Images Added")
        if "output_directory" in ctx.results:
            outputs.str("output_directory", label="Output Directory")
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
            generated_msg = ""
            if ctx.results.get("generated_samples", 0) > 0:
                generated_msg = f" Added {ctx.results.get('generated_samples')} generated image(s) as new samples."
            outputs.str(
                "success_msg",
                label="Success",
                default=f"Successfully processed {ctx.results.get('processed')} sample(s) with chat completions. Check the '{ctx.params.get('result_field', 'chat_response')}' field in your samples.{generated_msg}",
                view=types.Notice(variant="success"),
            )

        return types.Property(
            outputs, view=types.View(label="Chat Completions Results")
        )
