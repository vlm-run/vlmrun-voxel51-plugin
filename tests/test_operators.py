"""
Simple tests for VLM Run plugin operators.
These tests verify basic operator initialization and configuration.
"""

import pytest
import sys
import os

# Add parent directory to path to import the plugin
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestVideoOperators:
    """Test video transcription operator."""

    def test_transcribe_video_init(self):
        """Test VLMRunTranscribeVideo operator initialization."""
        from __init__ import VLMRunTranscribeVideo

        operator = VLMRunTranscribeVideo()
        config = operator.config

        assert config.name == "vlmrun_transcribe_video"
        assert config.label == "VLM Run: Transcribe Video"
        assert config.dynamic is True
        assert operator is not None


class TestImageOperators:
    """Test image processing operators."""

    def test_caption_images_init(self):
        """Test VLMRunCaptionImages operator initialization."""
        from __init__ import VLMRunCaptionImages

        operator = VLMRunCaptionImages()
        config = operator.config

        assert config.name == "vlmrun_caption_images"
        assert config.label == "VLM Run: Caption Images"
        assert operator is not None

    def test_object_detection_init(self):
        """Test VLMRunObjectDetection operator initialization."""
        from __init__ import VLMRunObjectDetection

        operator = VLMRunObjectDetection()
        config = operator.config

        assert config.name == "vlmrun_object_detection"
        assert config.label == "VLM Run: Object Detection"
        assert operator is not None

    def test_person_detection_init(self):
        """Test VLMRunPersonDetection operator initialization."""
        from __init__ import VLMRunPersonDetection

        operator = VLMRunPersonDetection()
        config = operator.config

        assert config.name == "vlmrun_person_detection"
        assert config.label == "VLM Run: Person Detection"
        assert operator is not None


class TestDocumentOperators:
    """Test document processing operators."""

    def test_parse_invoices_init(self):
        """Test VLMRunParseInvoices operator initialization."""
        from __init__ import VLMRunParseInvoices

        operator = VLMRunParseInvoices()
        config = operator.config

        assert config.name == "vlmrun_parse_invoices"
        assert config.label == "VLM Run: Parse Invoices"
        assert operator is not None

    def test_layout_detection_init(self):
        """Test VLMRunLayoutDetection operator initialization."""
        from __init__ import VLMRunLayoutDetection

        operator = VLMRunLayoutDetection()
        config = operator.config

        assert config.name == "vlmrun_layout_detection"
        assert config.label == "VLM Run: Document Layout Detection"
        assert operator is not None


class TestChatCompletionsOperator:
    """Test chat completions operator."""

    def test_chat_completions_init(self):
        """Test VLMRunChatCompletions operator initialization."""
        from __init__ import VLMRunChatCompletions

        operator = VLMRunChatCompletions()
        config = operator.config

        assert config.name == "vlmrun_chat_completions"
        assert config.label == "VLM Run: Chat Completions (Orion)"
        assert config.dynamic is True
        assert operator is not None

    def test_chat_completions_has_required_methods(self):
        """Test VLMRunChatCompletions has all required operator methods."""
        from __init__ import VLMRunChatCompletions

        operator = VLMRunChatCompletions()

        assert hasattr(operator, 'config')
        assert hasattr(operator, 'resolve_input')
        assert hasattr(operator, 'execute')
        assert hasattr(operator, 'resolve_output')

    def test_chat_completions_media_content_helper(self):
        """Test the media content helper method exists."""
        from __init__ import VLMRunChatCompletions

        operator = VLMRunChatCompletions()

        assert hasattr(operator, '_get_media_content')
        assert hasattr(operator, '_process_chat_result')


class TestOperatorRegistry:
    """Test that all operators are properly registered."""

    def test_all_operators_imported(self):
        """Test that all operators can be imported."""
        from __init__ import (
            VLMRunTranscribeVideo,
            VLMRunCaptionImages,
            VLMRunObjectDetection,
            VLMRunPersonDetection,
            VLMRunParseInvoices,
            VLMRunLayoutDetection,
            VLMRunChatCompletions
        )

        # Verify all 7 operators are importable
        operators = [
            VLMRunTranscribeVideo,
            VLMRunCaptionImages,
            VLMRunObjectDetection,
            VLMRunPersonDetection,
            VLMRunParseInvoices,
            VLMRunLayoutDetection,
            VLMRunChatCompletions
        ]

        assert len(operators) == 7

        for op_class in operators:
            operator = op_class()
            assert operator is not None
            assert hasattr(operator, 'config')
            assert hasattr(operator, 'execute')