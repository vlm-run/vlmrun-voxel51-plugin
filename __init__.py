"""
VLM Run FiftyOne Plugin.

Integration plugin for VLM Run (vlm.run) with FiftyOne.
"""
# pylint: disable=no-member,no-name-in-module

import os
import os.path
import time
from pathlib import Path

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
import fiftyone.core.utils as fou
import fiftyone.core.labels as fol
from vlmrun.client.types import GenerationConfig

# Configuration constants
DEFAULT_API_URL = "https://api.vlm.run/v1"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_WAIT = 600  # 10 minutes
DEFAULT_POLL_INTERVAL = 5  # seconds
MAX_ERROR_DETAILS = 5

# Supported file extensions
VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
)

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

DOCUMENT_EXTENSIONS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
)

# Import all operators from their respective modules
# Handle both package imports and direct module imports for testing
import sys
if __name__ == "__main__" or "test" in sys.argv[0]:
    # Direct import for testing
    from video_transcription import VLMRunTranscribeVideo
    from image_captioning import VLMRunCaptionImages
    from invoice_parsing import VLMRunParseInvoices
    from object_detection import VLMRunObjectDetection
    from person_detection import VLMRunPersonDetection
    from layout_detection import VLMRunLayoutDetection
else:
    # Package imports for normal use
    try:
        from .video_transcription import VLMRunTranscribeVideo
        from .image_captioning import VLMRunCaptionImages
        from .invoice_parsing import VLMRunParseInvoices
        from .object_detection import VLMRunObjectDetection
        from .person_detection import VLMRunPersonDetection
        from .layout_detection import VLMRunLayoutDetection
    except ImportError:
        # Fallback to direct imports
        from video_transcription import VLMRunTranscribeVideo
        from image_captioning import VLMRunCaptionImages
        from invoice_parsing import VLMRunParseInvoices
        from object_detection import VLMRunObjectDetection
        from person_detection import VLMRunPersonDetection
        from layout_detection import VLMRunLayoutDetection


def register(plugin):
    """Register all VLM Run operators."""
    plugin.register(VLMRunTranscribeVideo)
    plugin.register(VLMRunCaptionImages)
    plugin.register(VLMRunParseInvoices)
    plugin.register(VLMRunObjectDetection)
    plugin.register(VLMRunPersonDetection)
    plugin.register(VLMRunLayoutDetection)