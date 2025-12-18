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

        # Mode selection - analyze existing media or generate new content
        mode_choices = types.RadioGroup()
        mode_choices.add_choice("analyze", label="Analyze existing media")
        mode_choices.add_choice("generate", label="Generate new content (no input media)")
        inputs.enum(
            "mode",
            mode_choices.values(),
            default="analyze",
            label="Mode",
            description="Choose whether to analyze existing samples or generate new content",
            view=mode_choices,
        )

        mode = ctx.params.get("mode", "analyze")

        # Show notice if samples are selected (analyze mode only)
        if mode == "analyze" and ctx.selected:
            inputs.str(
                "selected_notice",
                view=types.Notice(
                    label=f"{len(ctx.selected)} sample(s) selected - only these will be processed"
                ),
            )

        # Target selection (only for analyze mode when no samples selected)
        has_view = ctx.dataset is not None and ctx.view != ctx.dataset.view()
        if mode == "analyze" and has_view and not ctx.selected:
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

        # Prompt input - different defaults based on mode
        if mode == "generate":
            inputs.str(
                "prompt",
                label="Prompt",
                description="Describe what you want to generate (e.g., 'Create a 5-second video of a sunset')",
                default="Create a short video clip",
                required=True,
            )

            # Number of generations
            inputs.int(
                "num_generations",
                label="Number of Generations",
                description="How many items to generate",
                default=1,
                required=True,
            )
        else:
            inputs.str(
                "prompt",
                label="Prompt",
                description="Natural language prompt for media analysis (e.g., 'Describe what you see')",
                default="Describe what you see in detail.",
                required=True,
            )

            # Response format selection (only for analyze mode)
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

            # Result field name (only for analyze mode)
            default_field = "chat_response"
            inputs.str(
                "result_field",
                label="Result Field",
                description="Field name to store the chat completion response",
                default=default_field,
                required=True,
            )

        # Temperature
        default_temp = 0.7 if mode == "generate" else 0.0
        inputs.float(
            "temperature",
            label="Temperature",
            description="Controls randomness (0 = deterministic, 1 = creative).",
            default=default_temp,
            required=False,
        )

        # System prompt (optional)
        default_system = "You are a content generation assistant. Always return an asset." if mode == "generate" else ""
        inputs.str(
            "system_prompt",
            label="System Prompt (Optional)",
            description="Optional system prompt to set context for the model",
            default=default_system,
            required=False,
        )

        if mode == "analyze":
            # Max samples (for testing/limiting)
            inputs.int(
                "max_samples",
                label="Max Samples (Optional)",
                description="Limit number of samples to process (leave empty for all)",
                required=False,
            )

            # Option to save generated artifacts as new samples
            inputs.bool(
                "save_generated_artifacts",
                label="Save Generated Artifacts",
                description="If enabled, any artifacts generated by the model (images, videos, etc.) will be saved and added as new samples to the dataset.",
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

        mode = ctx.params.get("mode", "analyze")

        # Branch based on mode
        if mode == "generate":
            return self._execute_generate(ctx, api_key)
        else:
            return self._execute_analyze(ctx, api_key)

    def _execute_generate(self, ctx: foo.ExecutionContext, api_key: str) -> Dict[str, Any]:
        """Execute content generation mode."""
        import re
        import shutil

        model = ctx.params.get("model", DEFAULT_MODEL)
        prompt = ctx.params["prompt"]
        temperature = ctx.params.get("temperature", 0.7)
        system_prompt = ctx.params.get("system_prompt")
        num_generations = ctx.params.get("num_generations", 1)

        # Initialize VLM Run client
        try:
            from vlmrun.client import VLMRun
        except ImportError:
            return {
                "error": "VLMRun package not installed. Run: fiftyone plugins requirements @vlm-run/vlmrun-voxel51-plugin --install"
            }

        # Get configuration
        api_url = os.getenv("VLMRUN_AGENT_API_URL", DEFAULT_AGENT_API_URL)
        timeout = float(os.getenv("VLMRUN_TIMEOUT", str(DEFAULT_TIMEOUT)))
        max_retries = int(os.getenv("VLMRUN_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))

        client = VLMRun(
            api_key=api_key,
            base_url=api_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Determine output directory
        if ctx.dataset and len(ctx.dataset) > 0:
            first_sample = ctx.dataset.first()
            if first_sample and first_sample.filepath:
                output_dir = Path(first_sample.filepath).parent / "generated_outputs"
            else:
                import tempfile
                output_dir = Path(tempfile.gettempdir()) / f"fiftyone_generated_{ctx.dataset.name}"
        else:
            import tempfile
            output_dir = Path(tempfile.gettempdir()) / "fiftyone_generated"

        output_dir.mkdir(parents=True, exist_ok=True)

        generated_count = 0
        errors = []

        for i in range(num_generations):
            try:
                # Build messages
                messages = []

                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                messages.append({"role": "user", "content": prompt})

                # Make the API call
                response = client.agent.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )

                # Extract artifacts from response
                content = response.choices[0].message.content or ""
                session_id = getattr(response, "session_id", None)

                if not session_id:
                    errors.append(f"Generation {i+1}: No session ID in response")
                    continue

                # Find all artifact references (including url_ type)
                artifact_refs = re.findall(r'(?:img|vid|aud|doc|url)_[a-zA-Z0-9]{6}', content)

                if not artifact_refs:
                    errors.append(f"Generation {i+1}: No artifacts in response. Response: {content[:200]}")
                    continue

                # Extension map for artifact types
                extension_map = {
                    "img": ".png",
                    "vid": ".mp4",
                    "aud": ".mp3",
                    "doc": ".pdf",
                }

                # Download and save each artifact
                for artifact_ref in artifact_refs:
                    try:
                        artifact = client.artifacts.get(
                            session_id=session_id,
                            object_id=artifact_ref,
                        )

                        artifact_type = artifact_ref.split("_")[0]

                        # For URL types, try to detect extension from the URL itself
                        if artifact_type == "url" and hasattr(artifact, "__str__"):
                            url_str = str(artifact)
                            from urllib.parse import urlparse
                            url_path = urlparse(url_str).path
                            extension = Path(url_path).suffix or ".bin"
                        else:
                            extension = extension_map.get(artifact_type, ".bin")

                        safe_prompt = re.sub(r'[^\w\s-]', '', prompt)[:30].strip().replace(' ', '_')
                        output_path = output_dir / f"{safe_prompt}_{artifact_ref}{extension}"

                        # Save the artifact based on its type
                        if isinstance(artifact, str):
                            if Path(artifact).exists():
                                # Local file path - copy it
                                shutil.copy2(artifact, output_path)
                            elif artifact.startswith(("http://", "https://")):
                                # URL - download it
                                import urllib.request
                                urllib.request.urlretrieve(str(artifact), str(output_path))
                            else:
                                errors.append(f"Unknown string artifact: {artifact[:100]}")
                                continue
                        elif isinstance(artifact, Path) and artifact.exists():
                            shutil.copy2(str(artifact), output_path)
                        elif hasattr(artifact, "save"):
                            artifact.save(str(output_path))
                        elif isinstance(artifact, bytes):
                            with open(output_path, "wb") as f:
                                f.write(artifact)
                        elif hasattr(artifact, "__str__") and str(artifact).startswith(("http://", "https://")):
                            # Pydantic URL type
                            import urllib.request
                            urllib.request.urlretrieve(str(artifact), str(output_path))
                        else:
                            errors.append(f"Unknown artifact type: {type(artifact)}")
                            continue

                        # Add as new sample to dataset
                        if ctx.dataset:
                            new_sample = fo.Sample(filepath=str(output_path))
                            new_sample.tags.append("vlmrun_generated")
                            new_sample["prompt"] = prompt
                            new_sample["generated_by"] = "vlmrun_chat_completions"
                            new_sample["model"] = model
                            ctx.dataset.add_sample(new_sample)

                        generated_count += 1

                    except Exception as e:
                        errors.append(f"Failed to save artifact {artifact_ref}: {str(e)}")

            except Exception as e:
                errors.append(f"Generation {i+1}: {str(e)}")

        # Refresh the app
        if not ctx.delegated and generated_count > 0:
            ctx.trigger("reload_dataset")

        result: Dict[str, Any] = {
            "generated": generated_count,
            "requested": num_generations,
            "output_directory": str(output_dir),
        }

        if errors:
            result["errors"] = len(errors)
            result["error_details"] = errors[:MAX_ERROR_DETAILS]

        return result

    def _execute_analyze(self, ctx: foo.ExecutionContext, api_key: str) -> Dict[str, Any]:
        """Execute media analysis mode."""
        target = ctx.params.get("target", "DATASET")
        model = ctx.params.get("model", DEFAULT_MODEL)
        prompt = ctx.params["prompt"]
        response_format = ctx.params.get("response_format", "text")
        result_field = ctx.params.get("result_field", "chat_response")
        temperature = ctx.params.get("temperature", 0.0)
        system_prompt = ctx.params.get("system_prompt")
        max_samples = ctx.params.get("max_samples")
        save_generated_artifacts = ctx.params.get("save_generated_artifacts", False)

        # Auto-determine output directory for generated artifacts
        output_dir = None
        if save_generated_artifacts:
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

        # Get samples - prioritize selected samples if any
        if ctx.selected:
            # Process only selected samples
            samples = ctx.dataset.select(ctx.selected)
        else:
            # Fall back to view or dataset
            sample_collection = ctx.view if target == "VIEW" else ctx.dataset

            # Apply max_samples limit if specified
            if max_samples and max_samples > 0:
                samples = sample_collection.take(max_samples)
            else:
                samples = sample_collection

        total_samples = len(samples)

        if total_samples == 0:
            return {
                "error": "No supported samples found. Select samples or ensure the collection has images, videos, or PDFs."
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

                    # Extract output artifacts and add as new samples if output_dir is provided
                    if output_dir:
                        saved_artifacts = self._extract_output_artifacts(
                            response, client, sample, output_dir, errors
                        )
                        for artifact_path in saved_artifacts:
                            # Create new sample for the generated artifact
                            new_sample = fo.Sample(filepath=artifact_path)
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

    def _extract_output_artifacts(
        self,
        response: Any,
        client: Any,
        source_sample: fo.Sample,
        output_dir: Path,
        errors: list,
    ) -> list:
        """Extract and save any output artifacts from the response.

        Detects artifact references (img_XXXXXX, vid_XXXXXX, etc.) in the response
        content, downloads them using the artifacts API, and saves them to disk.

        Args:
            response: The API response containing potential artifact refs.
            client: VLMRun client for downloading artifacts.
            source_sample: The original sample that generated this output.
            output_dir: Directory to save output artifacts.
            errors: List to append any errors to for reporting.

        Returns:
            List of file paths to saved artifacts.
        """
        import re
        import shutil

        saved_artifacts = []

        # Get response content
        if not hasattr(response, "choices") or not response.choices:
            return saved_artifacts

        content = response.choices[0].message.content or ""
        session_id = getattr(response, "session_id", None)

        if not session_id:
            return saved_artifacts

        # Find all artifact references (pattern: type_XXXXXX where type is img, vid, url, etc.)
        artifact_refs = re.findall(r'(?:img|vid|aud|doc|url)_[a-zA-Z0-9]{6}', content)

        if not artifact_refs:
            # Log what we searched through for debugging
            if content and len(content) > 0:
                errors.append(f"No artifacts found in response for {Path(source_sample.filepath).name}. Response preview: {content[:200]}...")
            return saved_artifacts

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Map artifact type prefixes to file extensions
        extension_map = {
            "img": ".png",
            "vid": ".mp4",
            "aud": ".mp3",
            "doc": ".pdf",
        }

        for artifact_ref in artifact_refs:
            try:
                # Download the artifact
                artifact = client.artifacts.get(
                    session_id=session_id,
                    object_id=artifact_ref,
                )

                # Determine file extension based on artifact type
                artifact_type = artifact_ref.split("_")[0]

                # For URL types, try to detect extension from the URL itself
                if artifact_type == "url" and hasattr(artifact, "__str__"):
                    url_str = str(artifact)
                    from urllib.parse import urlparse
                    url_path = urlparse(url_str).path
                    extension = Path(url_path).suffix or ".bin"
                else:
                    extension = extension_map.get(artifact_type, ".bin")

                # Generate filename based on source sample and artifact ID
                source_name = Path(source_sample.filepath).stem
                output_path = output_dir / f"{source_name}_{artifact_ref}{extension}"

                # Save the artifact - handle different types
                import urllib.request

                if isinstance(artifact, str):
                    if Path(artifact).exists():
                        # Artifact is a path to a cached file - copy it
                        shutil.copy2(artifact, output_path)
                    elif artifact.startswith(("http://", "https://")):
                        # URL - download it
                        urllib.request.urlretrieve(str(artifact), str(output_path))
                    else:
                        errors.append(f"Unknown string artifact: {artifact[:100]}")
                        continue
                elif isinstance(artifact, Path) and artifact.exists():
                    # Artifact is a Path object - copy it
                    shutil.copy2(str(artifact), output_path)
                elif hasattr(artifact, "save"):
                    # PIL Image or similar object with save method
                    artifact.save(str(output_path))
                elif isinstance(artifact, bytes):
                    # Raw bytes
                    with open(output_path, "wb") as f:
                        f.write(artifact)
                elif hasattr(artifact, "read"):
                    # File-like object
                    with open(output_path, "wb") as f:
                        f.write(artifact.read())
                elif hasattr(artifact, "content"):
                    # Response-like object with content attribute
                    with open(output_path, "wb") as f:
                        f.write(artifact.content)
                elif hasattr(artifact, "__str__") and str(artifact).startswith(("http://", "https://")):
                    # Pydantic URL type
                    urllib.request.urlretrieve(str(artifact), str(output_path))
                else:
                    errors.append(f"Unknown artifact type: {type(artifact)}")
                    continue

                saved_artifacts.append(str(output_path))

            except Exception as e:
                # Log artifact download errors for debugging
                errors.append(f"Failed to save artifact {artifact_ref}: {str(e)}")

        return saved_artifacts

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

        mode = ctx.params.get("mode", "analyze")

        # Show results based on mode
        if mode == "generate":
            # Generate mode results
            if "generated" in ctx.results:
                outputs.int("generated", label="Items Generated")
            if "requested" in ctx.results:
                outputs.int("requested", label="Items Requested")
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

            # Success message for generate mode
            if ctx.results.get("generated", 0) > 0:
                outputs.str(
                    "success_msg",
                    label="Success",
                    default=f"Successfully generated {ctx.results.get('generated')} item(s). They have been added to your dataset with the 'vlmrun_generated' tag.",
                    view=types.Notice(variant="success"),
                )
        else:
            # Analyze mode results
            if "processed" in ctx.results:
                outputs.int("processed", label="Samples Processed")
            if "total" in ctx.results:
                outputs.int("total", label="Total Samples")
            if "generated_samples" in ctx.results:
                outputs.int("generated_samples", label="Generated Artifacts Added")
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

            # Success message for analyze mode
            if ctx.results.get("processed", 0) > 0:
                generated_msg = ""
                if ctx.results.get("generated_samples", 0) > 0:
                    generated_msg = f" Added {ctx.results.get('generated_samples')} generated artifact(s) as new samples."
                outputs.str(
                    "success_msg",
                    label="Success",
                    default=f"Successfully processed {ctx.results.get('processed')} sample(s) with chat completions. Check the '{ctx.params.get('result_field', 'chat_response')}' field in your samples.{generated_msg}",
                    view=types.Notice(variant="success"),
                )

        return types.Property(
            outputs, view=types.View(label="Chat Completions Results")
        )
