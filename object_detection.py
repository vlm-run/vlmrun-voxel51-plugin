"""
Object Detection Operator

Detect objects in images using VLM Run's object detection domain.
"""

import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone as fo

from .utils import process_sample_with_vlm, get_api_key_from_ctx


class ObjectDetectionOperator(foo.Operator):
    """Operator for detecting objects in images using VLM Run."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="object_detection",
            label="VLM Run: Object Detection",
            dynamic=True,
        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
        
        # API Key input
        inputs.str(
            "api_key",
            label="VLM Run API Key",
            description="Your VLM Run API key (or set VLM_API_KEY environment variable)",
            required=False,
        )
        
        # Grounding option
        inputs.bool(
            "grounding",
            label="Enable Visual Grounding",
            description="Include bounding box coordinates in the results",
            default=True,
        )
        
        # Output field
        inputs.str(
            "output_field",
            label="Output Field Name",
            description="Field name to store detection results",
            default="vlmrun_detections",
        )
        
        # Execution mode
        inputs.enum(
            "delegate",
            types.Dropdown(label="Execution mode", default=False),
            label="Delegate execution",
            description="Whether to delegate execution to an orchestrator",
            values=[True, False],
        )
        
        return types.Property(inputs)
    
    def execute(self, ctx):
        """Execute the object detection operator."""
        # Get parameters
        api_key = get_api_key_from_ctx(ctx)
        if not api_key:
            raise ValueError(
                "VLM Run API key required. Provide via api_key parameter or VLM_API_KEY environment variable."
            )
        
        grounding = ctx.params.get("grounding", True)
        output_field = ctx.params.get("output_field", "vlmrun_detections")
        
        # Get selected samples or all samples in view
        view = ctx.view
        
        # Process each sample
        for sample in view:
            if sample.filepath and hasattr(sample, "media_type") and sample.media_type == "image":
                process_sample_with_vlm(
                    sample,
                    domain="image.object-detection",
                    api_key=api_key,
                    output_field=output_field,
                    grounding=grounding
                )
        
        ctx.ops.reload_dataset()
        
    def resolve_output(self, ctx):
        """Resolve output after execution."""
        outputs = types.Object()
        outputs.str("message", label="Status", view=types.Notice(label="Success"))
        return types.Property(outputs)
