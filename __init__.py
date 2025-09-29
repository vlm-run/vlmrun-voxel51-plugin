"""
VLM Run Plugin for FiftyOne

🎯 Extract structured data from visual and audio sources including documents, images, and videos

This plugin provides operators for extracting structured data from visual
sources using VLM Run's vision-language model.
"""

import fiftyone.operators as foo
from fiftyone.operators import types

# Import individual operator modules
from .object_detection import ObjectDetectionOperator
from .person_detection import PersonDetectionOperator
from .document_analysis import DocumentAnalysisOperator
from .invoice_parsing import InvoiceParsingOperator
from .layout_detection import LayoutDetectionOperator
from .video_transcription import VideoTranscriptionOperator


def register(plugin):
    """Register all operators with the plugin."""
    plugin.register(ObjectDetectionOperator)
    plugin.register(PersonDetectionOperator)
    plugin.register(DocumentAnalysisOperator)
    plugin.register(InvoiceParsingOperator)
    plugin.register(LayoutDetectionOperator)
    plugin.register(VideoTranscriptionOperator)