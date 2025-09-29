"""
Layout Detection Operator

Analyze document layout and structure using VLM Run.
"""

import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone as fo

from .utils import process_sample_with_vlm, get_api_key_from_ctx


class LayoutDetectionOperator(foo.Operator):
    """Operator for detecting document layout structure using VLM Run."""
    
    @property
    def config(self):
        return foo.OperatorConfig(
            name="layout_detection",
            label="VLM Run: Layout Detection",
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
        
        inputs.bool(
            "grounding",
            label="Enable Visual Grounding",
            description="Include bounding box coordinates for layout elements",
            default=True,
        )
        
        inputs.str(
            "output_field",
            label="Output Field Name",
            description="Field name to store layout detection results",
            default="vlmrun_layout",
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
        """Execute the layout detection operator."""
        api_key = get_api_key_from_ctx(ctx)
        if not api_key:
            raise ValueError(
                "VLM Run API key required. Provide via api_key parameter or VLM_API_KEY environment variable."
            )
        
        grounding = ctx.params.get("grounding", True)
        output_field = ctx.params.get("output_field", "vlmrun_layout")
        
        view = ctx.view
        
        for sample in view:
            if sample.filepath:
                media_type = getattr(sample, "media_type", None)
                if media_type in ["image", "pdf"] or sample.filepath.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
                    process_sample_with_vlm(
                        sample,
                        domain="document.structure",
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
