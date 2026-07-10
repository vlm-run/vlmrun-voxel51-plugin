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
from typing import Any, Dict, List, Optional

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou

# Configuration constants
DEFAULT_AGENT_API_URL = "https://agent.vlm.run/v1"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 5
MAX_ERROR_DETAILS = 5

# All available toolsets (used by checkbox UI and parser)
ALL_TOOLSETS = ["core", "viz", "image", "video", "document", "web", "image-gen"]

# Short descriptions for each toolset (shown next to checkboxes)
TOOLSET_DESCRIPTIONS = {
    "core": "Analyze images, extract content, process video",
    "image": "Object detection, text recognition, segmentation, quality assessment",
    "image-gen": "Create, transform, or apply effects to images",
    "viz": "Bounding boxes, keypoints, and segmentation masks",
    "document": "Layout detection, text extraction, structured data",
    "video": "Frame sampling, trimming, segmentation, video generation",
    "web": "Search the web to augment responses with real-time data",
}

# Default model
DEFAULT_MODEL = "vlmrun-orion-2:auto"

# Available Orion models. Orion 2 (code-execution agents) is listed first and
# is the default; Orion 1 (tool-calling agents) is retained for backward
# compatibility. Both share the same OpenAI-compatible chat-completions
# response contract, so selecting either works with the same code paths.
ORION_MODELS = [
    ("vlmrun-orion-2:fast", "Orion 2 Fast - Optimized for simple tasks with speed"),
    ("vlmrun-orion-2:auto", "Orion 2 Auto - Automatically selects best model for task"),
    ("vlmrun-orion-2:pro", "Orion 2 Pro - Most capable for complex workflows"),
    ("vlmrun-orion-1:fast", "Orion 1 Fast - Optimized for simple tasks with speed"),
    ("vlmrun-orion-1:auto", "Orion 1 Auto - Automatically selects best model for task"),
    ("vlmrun-orion-1:pro", "Orion 1 Pro - Most capable for complex workflows"),
]


def _default_model() -> str:
    """Return the effective default model.

    Overridable via the ``VLMRUN_DEFAULT_MODEL`` environment variable so users
    can pin a model (e.g. an Orion 1 variant, or a pinned backend variant) —
    and opt out of the Orion 2 default — without editing code. Falls back to
    ``DEFAULT_MODEL``.
    """
    return os.getenv("VLMRUN_DEFAULT_MODEL") or DEFAULT_MODEL


# Long-running-request handling. Orion runs on Modal, whose HTTP gateway closes
# the connection after ~150s; a request that outlives that window returns
# 303 See Other with a Location URL that serves the result once the job finishes.
# The OpenAI client raises APIStatusError on the 303, so without this the plugin
# would record every slow Orion op (video/document generation, large images) as a
# failure even though it completed — and billed — server-side. Overridable via
# VLMRUN_MAX_WAIT / VLMRUN_POLL_INTERVAL.
DEFAULT_MAX_WAIT = 600.0
DEFAULT_POLL_INTERVAL = 3.0


def _poll_redirect(
    client: Any, location: str, max_wait: float, poll_interval: float
) -> Dict[str, Any]:
    """Poll a 303 redirect target until the Orion completion result is ready.

    The result URL long-polls (holds the connection for minutes) and, while the
    job runs, may answer 202/204/303 or an intermittent 5xx; a read timeout on
    the poll simply means "not done yet." Returns the parsed completion body.
    """
    import time
    from urllib.parse import urljoin

    import requests

    url = urljoin("https://agent.vlm.run", location)
    headers = {"Authorization": f"Bearer {client.api_key}"}
    deadline = time.monotonic() + max_wait
    consecutive_5xx = 0
    while True:
        try:
            resp = requests.get(url, headers=headers, timeout=120, allow_redirects=False)
        except (requests.Timeout, requests.ConnectionError):
            # long-poll held the connection past the read timeout — not done yet
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Orion long-running request did not complete within {max_wait}s"
                )
            continue
        if resp.status_code in (200, 201) and resp.content:  # completion body is ready
            return resp.json()
        if resp.status_code in (202, 204, 303) or resp.status_code >= 500:
            # 202/204/303: result not ready yet (204 = No Content is returned
            # repeatedly while the job runs). 5xx: the poll endpoint intermittently
            # errors mid-job; tolerate a bounded number in a row.
            if resp.status_code >= 500:
                consecutive_5xx += 1
                if consecutive_5xx > 5:
                    raise RuntimeError(
                        f"Polling Orion request failed with status {resp.status_code}: {resp.text[:300]}"
                    )
            else:
                consecutive_5xx = 0
                if resp.status_code == 303:
                    url = urljoin(url, resp.headers.get("location", url))
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Orion long-running request did not complete within {max_wait}s"
                )
            time.sleep(poll_interval)
            continue
        raise RuntimeError(
            f"Polling Orion request failed with status {resp.status_code}: {resp.text[:300]}"
        )


def _create_completion(client: Any, **kwargs: Any) -> Any:
    """Call ``client.agent.completions.create`` with transparent 303 handling.

    Orion returns 303 for any request that outlives the ~150s Modal window; the
    OpenAI client surfaces that as ``APIStatusError``. We catch it, poll the
    Location URL, and rebuild a typed ``ChatCompletion`` so callers keep using
    ``response.choices[...]`` and the ``session_id`` extra field unchanged.
    """
    import openai

    try:
        return client.agent.completions.create(**kwargs)
    except openai.APIStatusError as exc:
        location = exc.response.headers.get("location")
        if exc.response.status_code != 303 or not location:
            raise
        max_wait = float(os.getenv("VLMRUN_MAX_WAIT", str(DEFAULT_MAX_WAIT)))
        poll_interval = float(os.getenv("VLMRUN_POLL_INTERVAL", str(DEFAULT_POLL_INTERVAL)))
        result = _poll_redirect(client, location, max_wait, poll_interval)

        from openai.types.chat import ChatCompletion

        return ChatCompletion.model_validate(result)


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

# Video edit/tools (trim, sample, extract segments) return a URL to the edited
# video via structured output — NOT a downloadable artifact ref. See VLM Run
# docs: https://docs.vlm.run/agents/capabilities/video/tools
_VIDEO_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "URL of the edited/trimmed video"},
        "start_time": {"type": "string", "description": "Start time (HH:MM:SS.MS), if applicable"},
        "end_time": {"type": "string", "description": "End time (HH:MM:SS.MS), if applicable"},
    },
    "required": ["url"],
}

# --- Vendored artifact schema helpers (from fiftyone.utils.vlmrun) ---
# These are vendored here to avoid depending on an unreleased fiftyone util
# and to add UrlRef/ArrayRef support.

_SINGULAR_ARTIFACT_TYPES = {
    "image": "ImageRef",
    "video": "VideoRef",
    "audio": "AudioRef",
    "document": "DocumentRef",
    "url": "UrlRef",
    "array": "ArrayRef",
}

_PLURAL_ARTIFACT_TYPES = {
    "images": ("ImageRef", "List of images"),
    "videos": ("VideoRef", "List of videos"),
    "audios": ("AudioRef", "List of audio files"),
    "documents": ("DocumentRef", "List of documents"),
    "urls": ("UrlRef", "List of URLs"),
    "arrays": ("ArrayRef", "List of arrays"),
}


def _build_artifact_schema(output_artifacts):
    """Build a JSON schema for the requested artifact types.

    Args:
        output_artifacts: a list of artifact type strings (singular or plural).

    Returns:
        a JSON schema dict, or None if no valid artifact types found.
    """
    from pydantic import BaseModel, Field
    from vlmrun.types.refs import (
        ArrayRef,
        AudioRef,
        DocumentRef,
        ImageRef,
        UrlRef,
        VideoRef,
    )

    ref_classes = {
        "ImageRef": ImageRef,
        "VideoRef": VideoRef,
        "AudioRef": AudioRef,
        "DocumentRef": DocumentRef,
        "UrlRef": UrlRef,
        "ArrayRef": ArrayRef,
    }

    annotations = {}
    field_defaults = {}

    for artifact_type in output_artifacts:
        if artifact_type in _SINGULAR_ARTIFACT_TYPES:
            ref_name = _SINGULAR_ARTIFACT_TYPES[artifact_type]
            annotations[artifact_type] = ref_classes[ref_name]
        elif artifact_type in _PLURAL_ARTIFACT_TYPES:
            ref_name, description = _PLURAL_ARTIFACT_TYPES[artifact_type]
            ref_class = ref_classes[ref_name]
            annotations[artifact_type] = List[ref_class]
            field_defaults[artifact_type] = Field(
                ..., description=description
            )

    if not annotations:
        return None

    namespace = {"__annotations__": annotations}
    namespace.update(field_defaults)
    DynamicModel = type("ArtifactResponse", (BaseModel,), namespace)
    return DynamicModel.model_json_schema()


def _parse_artifact_ids(content_str, output_artifacts):
    """Parse artifact IDs from a JSON response string.

    Args:
        content_str: a JSON string from the agent response content.
        output_artifacts: a list of artifact type strings that were requested.

    Returns:
        a dict mapping artifact type names to IDs (str) or lists of IDs.
    """
    try:
        parsed = json.loads(content_str)
    except (json.JSONDecodeError, TypeError):
        return {}

    result = {}
    for artifact_type in output_artifacts:
        if artifact_type not in parsed:
            continue

        artifact_data = parsed[artifact_type]

        if isinstance(artifact_data, list):
            ids = []
            for item in artifact_data:
                if isinstance(item, dict) and "id" in item:
                    ids.append(item["id"])
                elif isinstance(item, str):
                    ids.append(item)
            result[artifact_type] = ids
        elif isinstance(artifact_data, dict) and "id" in artifact_data:
            result[artifact_type] = artifact_data["id"]
        elif isinstance(artifact_data, str):
            result[artifact_type] = artifact_data
        else:
            result[artifact_type] = artifact_data

    return result


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
            # Long Orion operations (video edit/generation, document redaction)
            # can exceed the synchronous execution limit; allow users to run
            # them as delegated (background) jobs. The operator already handles
            # ctx.delegated throughout execute().
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

        # Detect dataset media type for filtering compatible modes/toolsets
        media_type = None
        if ctx.dataset is not None:
            media_type = ctx.dataset.media_type

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

        # Mode selection — filter by media type compatibility
        # For unknown/None media types, show all options
        show_all = media_type in (None, "unknown")
        mode_choices = types.RadioGroup()
        mode_choices.add_choice("analyze", label="Analyze existing media")
        if show_all or media_type != "video":
            mode_choices.add_choice("annotate", label="Annotate media (detections, keypoints, segmentation)")
        mode_choices.add_choice("edit", label="Edit existing media")
        if show_all or media_type in ("image", "video"):
            mode_choices.add_choice("generate", label="Generate new content (no input media)")
        inputs.enum(
            "mode",
            mode_choices.values(),
            default="analyze",
            label="Mode",
            description="Choose whether to analyze, edit, or generate content",
            view=mode_choices,
        )

        mode = ctx.params.get("mode", "analyze")

        # Show notice if samples are selected
        if mode in ("analyze", "annotate", "edit") and ctx.selected:
            inputs.str(
                "selected_notice",
                view=types.Notice(
                    label=f"{len(ctx.selected)} sample(s) selected - only these will be processed"
                ),
            )
        elif mode == "generate" and ctx.selected:
            inputs.str(
                "selected_notice",
                view=types.Notice(
                    label=f"{len(ctx.selected)} sample(s) selected as reference media"
                ),
            )
            inputs.bool(
                "use_reference_media",
                label="Use selected samples as reference",
                description="Attach the selected media as input for image-to-image, inpainting, or style transfer workflows",
                default=True,
            )

        # Target selection (for analyze/edit/annotate modes when no samples selected)
        has_view = ctx.dataset is not None and ctx.view != ctx.dataset.view()
        if mode in ("analyze", "annotate", "edit") and has_view and not ctx.selected:
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

        # Honor a VLMRUN_DEFAULT_MODEL override; if it names a model not in the
        # built-in list (e.g. a pinned backend variant), expose it as a choice
        # so the dropdown default stays valid.
        default_model = _default_model()
        if default_model not in model_choices.values():
            model_choices.add_choice(
                default_model,
                label=f"{default_model} (from VLMRUN_DEFAULT_MODEL)",
            )

        inputs.enum(
            "model",
            model_choices.values(),
            default=default_model,
            label="Model",
            description="Select the Orion model variant to use",
            view=model_choices,
        )

        # Prompt input - different defaults based on mode
        if mode == "generate":
            if media_type == "video":
                gen_default_prompt = "Create a short video clip of a sunset over the ocean"
            else:
                gen_default_prompt = "A photorealistic image of a sunset over the ocean"

            inputs.str(
                "prompt",
                label="Prompt",
                description="Describe what you want to generate",
                default=gen_default_prompt,
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
        elif mode == "annotate":
            # Output type for annotate mode
            output_type_choices = types.Dropdown()
            output_type_choices.add_choice("detections", label="Detections (bounding boxes)")
            output_type_choices.add_choice("keypoints", label="Keypoints")
            output_type_choices.add_choice("segmentation", label="Segmentation masks")

            inputs.enum(
                "output_type",
                output_type_choices.values(),
                default="detections",
                label="Output Type",
                description="Type of spatial annotation to produce",
                view=output_type_choices,
            )

            inputs.str(
                "prompt",
                label="Prompt",
                description="Describe what to detect/annotate (e.g., 'Detect all people and vehicles', 'Find keypoints on faces')",
                default="Detect all objects in the image",
                required=True,
            )

            output_type = ctx.params.get("output_type", "detections")
            default_annotation_field = f"vlmrun_{output_type}"

            inputs.str(
                "result_field",
                label="Result Field",
                description="Field name to store the annotations",
                default=default_annotation_field,
                required=True,
            )
        elif mode == "edit":
            inputs.str(
                "prompt",
                label="Prompt",
                description="Describe the edit to apply",
                default="Blur all faces in the video" if media_type == "video" else "Blur all faces in the image",
                required=True,
            )

            # Result field name for edit mode
            inputs.str(
                "result_field",
                label="Result Field",
                description="Field name to store the filepath to the edited media",
                default="edited_media",
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
        if mode == "generate":
            default_system = "You are a content generation assistant. Always return an asset."
        elif mode == "edit":
            default_system = (
                "You are a video editing assistant. Return the edited video as a URL."
                if media_type == "video"
                else "You are an image editing assistant. Always return the edited image as an artifact."
            )
        elif mode == "annotate":
            default_system = ""
        else:
            default_system = ""
        inputs.str(
            "system_prompt",
            label="System Prompt (Optional)",
            description="Optional system prompt to set context for the model",
            default=default_system,
            required=False,
        )

        # Toolsets — checkboxes filtered by mode and media type
        # For unknown/None media types, show all available toolsets
        if mode == "generate":
            toolsets_default = "video" if media_type == "video" else "image-gen"
            if show_all:
                available_toolsets = ["image-gen", "video", "core", "web"]
            elif media_type == "video":
                available_toolsets = ["video", "core", "web"]
            else:
                available_toolsets = ["image-gen", "video", "core", "web"]
        elif mode == "edit":
            toolsets_default = "video" if media_type == "video" else "image-gen"
            if show_all:
                available_toolsets = ["image-gen", "video", "image", "document", "core", "web"]
            elif media_type == "video":
                available_toolsets = ["video", "core", "web"]
            else:
                available_toolsets = ["image-gen", "image", "document", "core", "web"]
        elif mode == "annotate":
            toolsets_default = "viz"
            available_toolsets = ["viz", "core", "image", "document"]
        else:
            # Analyze mode
            toolsets_default = "video" if media_type == "video" else "core"
            if show_all:
                available_toolsets = ["core", "image", "video", "document", "web"]
            elif media_type == "video":
                available_toolsets = ["core", "video", "web"]
            else:
                available_toolsets = ["core", "image", "document", "web"]

        default_toolsets = [toolsets_default]

        inputs.view(
            "toolsets_header",
            types.Header(
                label="Toolsets",
                description="Select which toolsets to enable",
            ),
        )
        for ts in available_toolsets:
            desc = TOOLSET_DESCRIPTIONS.get(ts, "")
            inputs.bool(
                f"toolset_{ts.replace('-', '_')}",
                label=f"{ts} — {desc}" if desc else ts,
                default=(ts in default_toolsets),
                view=types.CheckboxView(),
            )

        if mode in ("analyze", "annotate", "edit"):
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

        mode = ctx.params.get("mode", "analyze")

        # Branch based on mode
        if mode == "generate":
            return self._execute_generate(ctx, api_key)
        elif mode == "edit":
            return self._execute_edit(ctx, api_key)
        elif mode == "annotate":
            return self._execute_annotate(ctx, api_key)
        else:
            return self._execute_analyze(ctx, api_key)

    @staticmethod
    def _parse_toolsets(ctx, default: str) -> List[str]:
        """Parse toolsets from checkbox boolean parameters.

        Args:
            ctx: The execution context.
            default: Default toolset string (e.g. "core").

        Returns:
            List of toolset strings.
        """
        selected = []
        for ts in ALL_TOOLSETS:
            param_name = f"toolset_{ts.replace('-', '_')}"
            val = ctx.params.get(param_name)
            if val is not None:
                if val:
                    selected.append(ts)
            # If param doesn't exist (not shown for this media type), skip
        return selected if selected else [default]

    @staticmethod
    def _build_extra_body(ctx, toolsets: List[str]) -> Dict[str, Any]:
        """Build the extra_body dict with toolsets and metadata.

        Args:
            ctx: The execution context.
            toolsets: List of toolset strings.

        Returns:
            Dict for the extra_body API parameter.
        """
        extra_body: Dict[str, Any] = {"toolsets": toolsets}
        return extra_body

    def _execute_annotate(self, ctx: foo.ExecutionContext, api_key: str) -> Dict[str, Any]:
        """Execute annotation mode — produces FiftyOne spatial labels.

        Supports detections (bounding boxes), keypoints, and segmentation masks.
        Uses the 'viz' toolset and structured JSON output to get spatial data,
        then converts to native FiftyOne label types.
        """
        target = ctx.params.get("target", "DATASET")
        model = ctx.params.get("model", _default_model())
        prompt = ctx.params["prompt"]
        output_type = ctx.params.get("output_type", "detections")
        result_field = ctx.params.get("result_field", "vlmrun_annotations")
        temperature = ctx.params.get("temperature", 0.0)
        system_prompt = ctx.params.get("system_prompt")
        max_samples = ctx.params.get("max_samples")
        toolsets = self._parse_toolsets(ctx, "viz")

        try:
            from vlmrun.client import VLMRun
        except ImportError:
            return {
                "error": "VLMRun package not installed. Run: fiftyone plugins requirements @vlm-run/vlmrun-voxel51-plugin --install"
            }

        api_url = os.getenv("VLMRUN_AGENT_API_URL", DEFAULT_AGENT_API_URL)
        timeout = float(os.getenv("VLMRUN_TIMEOUT", str(DEFAULT_TIMEOUT)))
        max_retries = int(os.getenv("VLMRUN_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))

        client = VLMRun(
            api_key=api_key,
            base_url=api_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Get samples
        if ctx.selected:
            samples = ctx.dataset.select(ctx.selected)
        else:
            sample_collection = ctx.view if target == "VIEW" else ctx.dataset
            if max_samples and max_samples > 0:
                samples = sample_collection.take(max_samples)
            else:
                samples = sample_collection

        total_samples = len(samples)
        if total_samples == 0:
            return {"error": "No samples found to annotate."}

        # Build the structured output schema based on output_type
        annotation_schema = self._build_annotation_schema(output_type)

        # Build the system prompt that instructs the model to return structured annotations
        annotation_system_prompt = self._build_annotation_system_prompt(output_type)
        if system_prompt:
            annotation_system_prompt = f"{annotation_system_prompt}\n\n{system_prompt}"

        processed = 0
        errors = []

        with fou.ProgressBar(total=total_samples) as pb:
            for sample in samples:
                try:
                    if not sample.filepath.lower().endswith(SUPPORTED_EXTENSIONS):
                        pb.update()
                        continue

                    file_path = Path(sample.filepath)
                    media_content = self._get_media_content(file_path, client)

                    messages = []
                    if annotation_system_prompt:
                        messages.append({"role": "system", "content": annotation_system_prompt})

                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            media_content,
                        ],
                    })

                    request_kwargs: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "extra_body": self._build_extra_body(ctx, toolsets),
                    }

                    if annotation_schema:
                        request_kwargs["response_format"] = {
                            "type": "json_schema",
                            "schema": annotation_schema,
                        }
                    else:
                        request_kwargs["response_format"] = {"type": "json_object"}

                    response = _create_completion(client, **request_kwargs)

                    content = response.choices[0].message.content or ""

                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        errors.append(
                            f"Failed to parse JSON for {Path(sample.filepath).name}: {content[:200]}"
                        )
                        pb.update()
                        continue

                    # Convert parsed JSON to FiftyOne label types
                    label = self._convert_annotations(parsed, output_type, errors, sample)
                    if label is not None:
                        sample[result_field] = label
                        sample.save()
                        processed += 1
                    else:
                        errors.append(
                            f"No {output_type} found for {os.path.basename(sample.filepath)}. "
                            f"The model returned valid JSON but with an empty '{output_type}' list. "
                            f"Try a more specific prompt or a different sample."
                        )

                except Exception as e:
                    errors.append(f"Failed to annotate {os.path.basename(sample.filepath)}: {str(e)}")

                pb.update()

        if not ctx.delegated:
            ctx.trigger("reload_dataset")

        result: Dict[str, Any] = {
            "processed": processed,
            "total": total_samples,
            "errors": len(errors),
        }

        if errors:
            result["error_details"] = errors[:MAX_ERROR_DETAILS]

        return result

    def _build_annotation_schema(self, output_type: str) -> Optional[Dict]:
        """Build a JSON schema for the annotation output type.

        Args:
            output_type: One of "detections", "keypoints", "segmentation".

        Returns:
            A JSON schema dict for the response_format, or None.
        """
        from pydantic import BaseModel, Field

        if output_type == "detections":
            class Detection(BaseModel):
                label: str = Field(..., description="Object class label")
                x: float = Field(..., description="Normalized x of top-left corner (0-1)")
                y: float = Field(..., description="Normalized y of top-left corner (0-1)")
                w: float = Field(..., description="Normalized width (0-1)")
                h: float = Field(..., description="Normalized height (0-1)")
                confidence: float = Field(1.0, description="Confidence score (0-1)")

            class AnnotationResponse(BaseModel):
                detections: List[Detection] = Field(..., description="List of detected objects")

            return AnnotationResponse.model_json_schema()

        elif output_type == "keypoints":
            class Keypoint(BaseModel):
                label: str = Field(..., description="Keypoint group label")
                points: List[List[float]] = Field(
                    ..., description="List of [x, y] normalized coordinates (0-1)"
                )
                confidence: Optional[float] = Field(None, description="Confidence score (0-1)")

            class AnnotationResponse(BaseModel):
                keypoints: List[Keypoint] = Field(..., description="List of keypoint groups")

            return AnnotationResponse.model_json_schema()

        elif output_type == "segmentation":
            class Segment(BaseModel):
                label: str = Field(..., description="Segment class label")
                mask_url: Optional[str] = Field(None, description="URL to mask PNG image")
                polygon: Optional[List[List[float]]] = Field(
                    None, description="Polygon as list of [x, y] normalized points (0-1)"
                )

            class AnnotationResponse(BaseModel):
                segments: List[Segment] = Field(..., description="List of segmentation regions")

            return AnnotationResponse.model_json_schema()

        return None

    def _build_annotation_system_prompt(self, output_type: str) -> str:
        """Build a system prompt tailored to the annotation output type."""
        if output_type == "detections":
            return (
                "You are an object detection model. Return a JSON object with a 'detections' array. "
                "Each detection has: label (string), x, y, w, h (all floats 0-1 representing "
                "normalized bounding box as top-left x, top-left y, width, height), and confidence (float 0-1)."
            )
        elif output_type == "keypoints":
            return (
                "You are a keypoint detection model. Return a JSON object with a 'keypoints' array. "
                "Each keypoint group has: label (string), points (array of [x, y] pairs, "
                "normalized 0-1), and optionally confidence (float 0-1)."
            )
        elif output_type == "segmentation":
            return (
                "You are a segmentation model. Return a JSON object with a 'segments' array. "
                "Each segment has: label (string), and either mask_url (URL to a PNG mask) "
                "or polygon (array of [x, y] normalized points 0-1 forming the boundary)."
            )
        return ""

    def _convert_annotations(
        self,
        parsed: Dict,
        output_type: str,
        errors: list,
        sample: fo.Sample,
    ) -> Optional[Any]:
        """Convert parsed JSON annotations to FiftyOne label types.

        Args:
            parsed: Parsed JSON dict from the model response.
            output_type: One of "detections", "keypoints", "segmentation".
            errors: Error list for reporting.
            sample: The current FiftyOne sample.

        Returns:
            A FiftyOne label object, or None on failure.
        """
        if output_type == "detections":
            detections_data = parsed.get("detections", [])
            if not detections_data:
                return None

            fo_detections = []
            for det in detections_data:
                try:
                    fo_detections.append(
                        fol.Detection(
                            label=det.get("label", "object"),
                            bounding_box=[
                                det.get("x", 0),
                                det.get("y", 0),
                                det.get("w", 0),
                                det.get("h", 0),
                            ],
                            confidence=det.get("confidence"),
                        )
                    )
                except Exception as e:
                    errors.append(f"Bad detection entry: {e}")

            return fol.Detections(detections=fo_detections) if fo_detections else None

        elif output_type == "keypoints":
            keypoints_data = parsed.get("keypoints", [])
            if not keypoints_data:
                return None

            fo_keypoints = []
            for kp in keypoints_data:
                try:
                    points = kp.get("points", [])
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    # FiftyOne Keypoint.confidence is a list (one per point)
                    conf = kp.get("confidence")
                    confidence = [conf] * len(points) if conf is not None else None
                    fo_keypoints.append(
                        fol.Keypoint(
                            label=kp.get("label", "keypoint"),
                            points=list(zip(xs, ys)),
                            confidence=confidence,
                        )
                    )
                except Exception as e:
                    errors.append(f"Bad keypoint entry: {e}")

            return fol.Keypoints(keypoints=fo_keypoints) if fo_keypoints else None

        elif output_type == "segmentation":
            segments_data = parsed.get("segments", [])
            if not segments_data:
                return None

            # For polygon-based segmentation, convert to FiftyOne Polylines
            fo_polylines = []
            for seg in segments_data:
                try:
                    polygon = seg.get("polygon")
                    if polygon:
                        fo_polylines.append(
                            fol.Polyline(
                                label=seg.get("label", "segment"),
                                points=[[(p[0], p[1]) for p in polygon]],
                                closed=True,
                                filled=True,
                            )
                        )
                    # mask_url handling could be added here in the future
                except Exception as e:
                    errors.append(f"Bad segment entry: {e}")

            return fol.Polylines(polylines=fo_polylines) if fo_polylines else None

        return None

    def _execute_generate(self, ctx: foo.ExecutionContext, api_key: str) -> Dict[str, Any]:
        """Execute content generation mode."""
        import re
        import shutil

        model = ctx.params.get("model", _default_model())
        prompt = ctx.params["prompt"]
        temperature = ctx.params.get("temperature", 0.7)
        system_prompt = ctx.params.get("system_prompt")
        num_generations = ctx.params.get("num_generations", 1)
        toolsets = self._parse_toolsets(ctx, "image-gen")
        use_reference_media = ctx.params.get("use_reference_media", False)

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

        # Collect reference media from selected samples (for image-to-image workflows)
        reference_media = []
        if use_reference_media and ctx.selected:
            for sample in ctx.dataset.select(ctx.selected):
                if sample.filepath and sample.filepath.lower().endswith(SUPPORTED_EXTENSIONS):
                    reference_media.append(Path(sample.filepath))

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

                # Attach reference media if available (image-to-image workflows)
                if reference_media:
                    user_content = [{"type": "text", "text": prompt}]
                    for ref_path in reference_media:
                        media_content = self._get_media_content(ref_path, client)
                        user_content.append(media_content)
                    messages.append({"role": "user", "content": user_content})
                else:
                    messages.append({"role": "user", "content": prompt})

                # Make the API call
                response = _create_completion(
                    client,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    extra_body=self._build_extra_body(ctx, toolsets),
                )

                # Extract artifacts from response
                content = response.choices[0].message.content or ""
                session_id = getattr(response, "session_id", None)

                if not session_id:
                    errors.append(f"Generation {i+1}: No session ID in response")
                    continue

                # Find all artifact references (including url_ types)
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
                        artifact = self._get_artifact_with_retry(
                            client=client,
                            session_id=session_id,
                            artifact_ref=artifact_ref,
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
                            try:
                                ctx.dataset.add_sample(new_sample)
                            except Exception as add_err:
                                # Media type mismatch (e.g. adding video to
                                # image-only dataset). The file is saved on
                                # disk; inform the user so they can load it
                                # into a compatible dataset.
                                errors.append(
                                    f"Artifact saved to {output_path} but could not be added "
                                    f"to the current dataset: {add_err}. "
                                    f"Try loading it into a dataset that supports this media type."
                                )

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

    def _execute_edit(self, ctx: foo.ExecutionContext, api_key: str) -> Dict[str, Any]:
        """Execute media editing mode.

        Uses the 'image' toolset to edit existing media and saves the
        modified artifacts back to the samples.
        """
        import re
        import shutil

        target = ctx.params.get("target", "DATASET")
        model = ctx.params.get("model", _default_model())
        prompt = ctx.params["prompt"]
        result_field = ctx.params.get("result_field", "edited_image")
        temperature = ctx.params.get("temperature", 0.0)
        system_prompt = ctx.params.get("system_prompt")
        max_samples = ctx.params.get("max_samples")
        toolsets = self._parse_toolsets(ctx, "image-gen")

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
        max_retries = int(
            os.getenv("VLMRUN_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))
        )

        client = VLMRun(
            api_key=api_key,
            base_url=api_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Determine output artifact types based on toolsets
        output_artifact_types = ["image"]
        if "video" in toolsets:
            output_artifact_types = ["video"]

        # Get samples
        if ctx.selected:
            samples = ctx.dataset.select(ctx.selected)
        else:
            sample_collection = ctx.view if target == "VIEW" else ctx.dataset
            if max_samples and max_samples > 0:
                samples = sample_collection.take(max_samples)
            else:
                samples = sample_collection

        total_samples = len(samples)

        if total_samples == 0:
            return {
                "error": "No samples found. Select samples or ensure the collection has images."
            }

        # Determine output directory
        first_sample = ctx.dataset.first()
        if first_sample and first_sample.filepath:
            output_dir = Path(first_sample.filepath).parent / "edited_outputs"
        else:
            import tempfile
            output_dir = Path(tempfile.gettempdir()) / f"fiftyone_edited_{ctx.dataset.name}"

        output_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        errors = []

        with fou.ProgressBar(total=total_samples) as pb:
            for sample in samples:
                try:
                    # Skip unsupported file types
                    if not sample.filepath.lower().endswith(SUPPORTED_EXTENSIONS):
                        pb.update()
                        continue

                    # Build messages with media
                    messages = []

                    if system_prompt:
                        messages.append({
                            "role": "system",
                            "content": system_prompt,
                        })

                    file_path = Path(sample.filepath)
                    media_content = self._get_media_content(file_path, client)

                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            media_content,
                        ],
                    })

                    # Video tools (trim/sample/segment) return a URL to the edited
                    # video via structured output; image edits return an artifact
                    # ref that must be downloaded via artifacts.get.
                    is_video = file_path.suffix.lower() in VIDEO_EXTENSIONS

                    request_kwargs: Dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "extra_body": self._build_extra_body(ctx, toolsets),
                    }

                    if is_video:
                        request_kwargs["response_format"] = {
                            "type": "json_schema",
                            "schema": _VIDEO_EDIT_SCHEMA,
                        }
                    else:
                        artifact_schema = _build_artifact_schema(output_artifact_types)
                        if artifact_schema:
                            request_kwargs["response_format"] = {
                                "type": "json_schema",
                                "schema": artifact_schema,
                            }

                    # Make the API call
                    response = _create_completion(client, **request_kwargs)
                    content = response.choices[0].message.content or ""
                    source_name = Path(sample.filepath).stem

                    if is_video:
                        # Video tools return a JSON pointer to the edited video in
                        # one of two forms — a direct URL, or an artifact id (e.g.
                        # vid_abc123, sometimes suffixed .mp4). The API picks which;
                        # handle both and download the bytes to a local file.
                        import urllib.request

                        pointer = None
                        try:
                            data = json.loads(content)
                            if isinstance(data, dict):
                                pointer = data.get("url") or data.get("video_url")
                        except (json.JSONDecodeError, TypeError):
                            pointer = None
                        if not pointer:
                            match = re.search(
                                r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*', content
                            )
                            pointer = match.group(0) if match else None
                        if not pointer:
                            match = re.search(r'(?:vid|img)_[a-zA-Z0-9]{6}', content)
                            pointer = match.group(0) if match else None

                        if not pointer:
                            errors.append(
                                f"No edited video URL or artifact returned for {Path(sample.filepath).name}. "
                                f"Response: {content[:200]}"
                            )
                            pb.update()
                            continue

                        output_path = output_dir / f"{source_name}_edited.mp4"

                        if pointer.startswith(("http://", "https://")):
                            # Form 1: direct download URL
                            urllib.request.urlretrieve(pointer, str(output_path))
                        else:
                            # Form 2: artifact id (strip any trailing extension,
                            # e.g. vid_abc123.mp4) -> exchange for bytes.
                            artifact_ref = pointer.split(".")[0]
                            session_id = getattr(response, "session_id", None)
                            if not session_id and hasattr(response, "model_extra"):
                                session_id = response.model_extra.get("session_id")
                            artifact = self._get_artifact_with_retry(
                                client=client,
                                session_id=session_id,
                                artifact_ref=artifact_ref,
                            )
                            if isinstance(artifact, Path) and artifact.exists():
                                shutil.copy2(str(artifact), output_path)
                            elif isinstance(artifact, str) and Path(artifact).exists():
                                shutil.copy2(artifact, output_path)
                            elif isinstance(artifact, str) and artifact.startswith(("http://", "https://")):
                                urllib.request.urlretrieve(artifact, str(output_path))
                            elif isinstance(artifact, bytes):
                                with open(output_path, "wb") as f:
                                    f.write(artifact)
                            else:
                                errors.append(
                                    f"Unknown video artifact type: {type(artifact)}"
                                )
                                pb.update()
                                continue
                    else:
                        session_id = getattr(response, "session_id", None)
                        if not session_id and hasattr(response, "model_extra"):
                            session_id = response.model_extra.get("session_id")

                        # Parse artifact IDs from content
                        artifact_ids = _parse_artifact_ids(content, output_artifact_types)
                        object_id = artifact_ids.get(output_artifact_types[0])

                        if not object_id:
                            refs = re.findall(r'(?:img|vid)_[a-zA-Z0-9]{6}', content)
                            if refs:
                                object_id = refs[0]

                        if not object_id:
                            errors.append(
                                f"No edited artifact produced for {Path(sample.filepath).name}. "
                                f"Response: {content[:200]}"
                            )
                            pb.update()
                            continue

                        # Download the artifact
                        artifact = self._get_artifact_with_retry(
                            client=client,
                            session_id=session_id,
                            artifact_ref=object_id,
                        )

                        artifact_type = object_id.split("_")[0] if "_" in object_id else "img"
                        ext_map = {"img": ".png", "vid": ".mp4", "aud": ".mp3", "doc": ".pdf"}
                        ext = ext_map.get(artifact_type, ".png")
                        output_path = output_dir / f"{source_name}_{object_id}{ext}"

                        if isinstance(artifact, str):
                            if Path(artifact).exists():
                                shutil.copy2(artifact, output_path)
                            elif artifact.startswith(("http://", "https://")):
                                import urllib.request
                                urllib.request.urlretrieve(str(artifact), str(output_path))
                            else:
                                errors.append(f"Unknown string artifact: {artifact[:100]}")
                                pb.update()
                                continue
                        elif isinstance(artifact, Path) and artifact.exists():
                            shutil.copy2(str(artifact), output_path)
                        elif hasattr(artifact, "save"):
                            artifact.save(str(output_path))
                        elif isinstance(artifact, bytes):
                            with open(output_path, "wb") as f:
                                f.write(artifact)
                        elif hasattr(artifact, "__str__") and str(artifact).startswith(("http://", "https://")):
                            import urllib.request
                            urllib.request.urlretrieve(str(artifact), str(output_path))
                        else:
                            errors.append(f"Unknown artifact type: {type(artifact)}")
                            pb.update()
                            continue

                    # Store filepath on the sample
                    sample[result_field] = str(output_path)
                    sample.save()

                    # Add edited image as a viewable sample
                    if ctx.dataset:
                        new_sample = fo.Sample(filepath=str(output_path))
                        new_sample.tags.append("vlmrun_edited")
                        new_sample["prompt"] = prompt
                        new_sample["source_filepath"] = sample.filepath
                        new_sample["generated_by"] = "vlmrun_chat_completions"
                        new_sample["model"] = model
                        ctx.dataset.add_sample(new_sample)

                    processed += 1

                except Exception as e:
                    error_msg = f"Failed to edit {os.path.basename(sample.filepath)}: {str(e)}"
                    errors.append(error_msg)

                pb.update()

        # Refresh the app
        if not ctx.delegated:
            ctx.trigger("reload_dataset")

        result: Dict[str, Any] = {
            "processed": processed,
            "total": total_samples,
            "output_directory": str(output_dir),
            "errors": len(errors),
        }

        if errors:
            result["error_details"] = errors[:MAX_ERROR_DETAILS]

        return result

    def _execute_analyze(self, ctx: foo.ExecutionContext, api_key: str) -> Dict[str, Any]:
        """Execute media analysis mode."""
        target = ctx.params.get("target", "DATASET")
        model = ctx.params.get("model", _default_model())
        prompt = ctx.params["prompt"]
        result_field = ctx.params.get("result_field", "chat_response")
        temperature = ctx.params.get("temperature", 0.0)
        system_prompt = ctx.params.get("system_prompt")
        max_samples = ctx.params.get("max_samples")
        toolsets = self._parse_toolsets(ctx, "core")

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
                        "extra_body": self._build_extra_body(ctx, toolsets),
                    }

                    # Make the API call using Orion agent completions
                    response = _create_completion(client, **request_kwargs)

                    # Extract and store the result
                    self._process_chat_result(
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

    def _get_artifact_with_retry(
        self,
        client: Any,
        session_id: str,
        artifact_ref: str,
        retry_count: int = 20,
        delay: float = 3.0,
    ) -> Any:
        """Fetch an artifact's bytes from VLM Run, retrying until it's available.

        Fetches ``GET /v1/artifacts`` directly instead of the SDK's
        ``client.artifacts.get()``. The SDK asserts the HTTP ``Content-Type``
        matches the artifact type and raises an ``AssertionError`` when VLM Run
        serves a mismatched type (e.g. an ``img_`` artifact sent as
        ``application/octet-stream``); fetching the raw bytes avoids that crash.
        Artifacts can also lag the completion response, so we poll with a delay.

        Args:
            client: VLMRun client instance (provides ``base_url`` and ``api_key``).
            session_id: The session ID from the response.
            artifact_ref: The artifact reference (e.g., ``img_abc123``); any
                trailing file extension is stripped.
            retry_count: Number of retries before giving up.
            delay: Delay in seconds between retries.

        Returns:
            Raw ``bytes`` for file artifacts (img/vid/doc/aud), or the URL string
            for ``url_`` artifacts (callers download it).

        Raises:
            RuntimeError: If retrieval fails after all retries, or on an
                unrecoverable auth error.
        """
        import time

        import requests

        object_id = artifact_ref.split(".")[0]  # strip any trailing extension
        obj_type = object_id.split("_")[0]
        url = f"{client.base_url.rstrip('/')}/artifacts"
        headers = {"Authorization": f"Bearer {client.api_key}"}
        params = {"object_id": object_id, "session_id": session_id}

        last_error: Any = None
        for attempt in range(retry_count):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=120)
            except Exception as e:  # transient network error — retry
                last_error = e
            else:
                if resp.status_code in (401, 403):
                    # auth failures will not resolve by waiting
                    raise RuntimeError(
                        f"Unauthorized fetching artifact {object_id} (HTTP {resp.status_code})"
                    )
                if resp.status_code == 200 and resp.content:
                    if obj_type == "url":
                        return resp.content.decode("utf-8", errors="replace").strip()
                    return resp.content
                last_error = RuntimeError(f"artifact not ready (HTTP {resp.status_code})")
            if attempt < retry_count - 1:
                time.sleep(delay)

        raise RuntimeError(
            f"Failed to get artifact {object_id} after {retry_count} retries: {last_error}"
        )

    def _process_chat_result(
        self,
        sample: fo.Sample,
        result: Any,
        result_field: str,
    ) -> None:
        """Process VLM Run chat completion result and update sample.

        Args:
            sample: The FiftyOne sample to update.
            result: The API response from VLM Run.
            result_field: The field name to store results.
        """
        # Extract the response content
        if hasattr(result, "choices") and result.choices:
            content = result.choices[0].message.content
        else:
            content = str(result)

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
        elif mode == "annotate":
            # Annotate mode results
            if "processed" in ctx.results:
                outputs.int("processed", label="Samples Annotated")
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

            if ctx.results.get("processed", 0) > 0:
                outputs.str(
                    "success_msg",
                    label="Success",
                    default=f"Successfully annotated {ctx.results.get('processed')} sample(s). Check the '{ctx.params.get('result_field', 'vlmrun_annotations')}' field for the annotations.",
                    view=types.Notice(variant="success"),
                )
        elif mode == "edit":
            # Edit mode results
            if "processed" in ctx.results:
                outputs.int("processed", label="Samples Edited")
            if "total" in ctx.results:
                outputs.int("total", label="Total Samples")
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

            # Success message for edit mode
            if ctx.results.get("processed", 0) > 0:
                outputs.str(
                    "success_msg",
                    label="Success",
                    default=f"Successfully edited {ctx.results.get('processed')} sample(s). Edited images are added to the dataset with the 'vlmrun_edited' tag. Filter by this tag to view them.",
                    view=types.Notice(variant="success"),
                )
        else:
            # Analyze mode results
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

            # Success message for analyze mode
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
