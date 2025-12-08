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
import os.path
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
            "json_object", label="JSON Object - Flexible JSON output"
        )
        response_format_choices.add_choice(
            "json_schema", label="JSON Schema - Strict schema enforcement"
        )

        inputs.enum(
            "response_format",
            response_format_choices.values(),
            default="text",
            label="Response Format",
            description="Choose how the model should format its response",
            view=response_format_choices,
        )

        # JSON Schema (only shown when json_schema is selected)
        response_format = ctx.params.get("response_format", "text")
        if response_format == "json_schema":
            inputs.str(
                "json_schema",
                label="JSON Schema",
                description="JSON schema for structured output (paste valid JSON schema)",
                default='{"type": "object", "properties": {"description": {"type": "string"}, "objects": {"type": "array", "items": {"type": "string"}}}, "required": ["description"]}',
                required=True,
            )
            inputs.str(
                "schema_name",
                label="Schema Name",
                description="Name for the schema (used in API request)",
                default="media_analysis",
                required=True,
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
        json_schema_str = ctx.params.get("json_schema")
        schema_name = ctx.params.get("schema_name", "media_analysis")
        max_samples = ctx.params.get("max_samples")

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

        # Parse JSON schema if provided
        json_schema: Optional[Dict[str, Any]] = None
        if response_format == "json_schema" and json_schema_str:
            try:
                json_schema = json.loads(json_schema_str)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON schema: {str(e)}"}

        processed = 0
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
                    elif response_format == "json_schema" and json_schema:
                        request_kwargs["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name,
                                "schema": json_schema
                            }
                        }

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
            response_format: The response format (text, json_object, json_schema).
        """
        # Extract the response content
        if hasattr(result, "choices") and result.choices:
            content = result.choices[0].message.content
        else:
            content = str(result)

        # Parse JSON responses
        if response_format in ("json_object", "json_schema"):
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
            usage_data: Dict[str, int] = {}
            if hasattr(result.usage, "prompt_tokens"):
                usage_data["prompt_tokens"] = result.usage.prompt_tokens
            if hasattr(result.usage, "completion_tokens"):
                usage_data["completion_tokens"] = result.usage.completion_tokens
            if hasattr(result.usage, "total_tokens"):
                usage_data["total_tokens"] = result.usage.total_tokens
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
                default=f"Successfully processed {ctx.results.get('processed')} sample(s) with chat completions. Check the '{ctx.params.get('result_field', 'chat_response')}' field in your samples.",
                view=types.Notice(variant="success"),
            )

        return types.Property(
            outputs, view=types.View(label="Chat Completions Results")
        )
