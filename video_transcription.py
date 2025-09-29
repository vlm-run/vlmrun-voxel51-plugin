"""
Video Transcription Operator

Transcribe audio from videos and perform comprehensive video analysis using VLM Run.
"""

import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone as fo

from .utils import process_sample_with_vlm, get_api_key_from_ctx


class VideoTranscriptionOperator(foo.Operator):
    """Operator for transcribing videos using VLM Run."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="video_transcription",
            label="VLM Run: Video Transcription",
            dynamic=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        inputs.str(
            "api_key",
            label="VLM Run API Key",
            description="Your VLM Run API key (or set VLM_API_KEY environment variable)",
            required=False,
        )
        
        inputs.enum(
            "analysis_mode",
            types.Dropdown(label="Analysis Mode"),
            label="Analysis Mode",
            description="Type of video analysis to perform",
            values=["transcription", "comprehensive", "objects", "scenes", "activities"],
            default="transcription",
        )
        
        inputs.str(
            "output_field",
            label="Output Field Name",
            description="Field name to store video analysis results",
            default="vlmrun_video",
        )
        
        inputs.enum(
            "delegate",
            types.Dropdown(label="Execution mode", default=False),
            label="Delegate execution",
            description="Whether to delegate execution to an orchestrator",
            values=[True, False],
        )
        
        return types.Property(inputs)
    
    def execute(self, ctx):
        """Execute the video transcription operator."""
        api_key = get_api_key_from_ctx(ctx)
        if not api_key:
            raise ValueError(
                "VLM Run API key required. Provide via api_key parameter or VLM_API_KEY environment variable."
            )
        
        analysis_mode = ctx.params.get("analysis_mode", "transcription")
        output_field = ctx.params.get("output_field", "vlmrun_video")
        
        view = ctx.view
        
        # Determine the domain based on analysis mode
        if analysis_mode == "transcription":
            domain = "video.transcription"
        else:
            domain = f"video.{analysis_mode}"
        
        for sample in view:
            if sample.filepath and hasattr(sample, "media_type") and sample.media_type == "video":
                process_sample_with_vlm(
                    sample,
                    domain=domain,
                    api_key=api_key,
                    output_field=output_field
                )
        
        ctx.ops.reload_dataset()
        
    def resolve_output(self, ctx):
        """Resolve output after execution."""
        outputs = types.Object()
        outputs.str("message", label="Status", view=types.Notice(label="Success"))
        return types.Property(outputs)
