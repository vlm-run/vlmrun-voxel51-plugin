"""
Simple tests for VLM Run plugin operators.
These tests verify basic operator initialization and configuration.

Integration tests require VLMRUN_API_KEY environment variable to be set.
Run with: pytest tests/test_operators.py -v -m integration
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path to import the plugin
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test samples directory
TEST_SAMPLES_DIR = Path(__file__).parent.parent / "test_samples"

# Skip integration tests if no API key
requires_api_key = pytest.mark.skipif(
    not os.getenv("VLMRUN_API_KEY"),
    reason="VLMRUN_API_KEY environment variable not set"
)


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


class TestOrionModelConfig:
    """Offline tests pinning the Orion model configuration.

    These guard the Orion 2 upgrade so CI fails if the default or the
    available-model list regress (the integration tests are skipped without
    an API key, so without these the constants are effectively untested).
    """

    def test_default_model_is_orion_2_auto(self):
        from chat_completions import DEFAULT_MODEL

        assert DEFAULT_MODEL == "vlmrun-orion-2:auto"

    def test_orion_2_variants_present_and_first(self):
        from chat_completions import ORION_MODELS

        model_ids = [model_id for model_id, _ in ORION_MODELS]

        # All three Orion 2 variants are offered...
        for variant in ("fast", "auto", "pro"):
            assert f"vlmrun-orion-2:{variant}" in model_ids

        # ...and listed before any Orion 1 variant (Orion 2 is the default family).
        first_orion_1 = next(
            i for i, m in enumerate(model_ids) if m.startswith("vlmrun-orion-1:")
        )
        last_orion_2 = max(
            i for i, m in enumerate(model_ids) if m.startswith("vlmrun-orion-2:")
        )
        assert last_orion_2 < first_orion_1

    def test_orion_1_retained_for_backward_compat(self):
        from chat_completions import ORION_MODELS

        model_ids = [model_id for model_id, _ in ORION_MODELS]
        for variant in ("fast", "auto", "pro"):
            assert f"vlmrun-orion-1:{variant}" in model_ids

    def test_default_model_is_selectable(self):
        from chat_completions import DEFAULT_MODEL, ORION_MODELS

        assert DEFAULT_MODEL in [model_id for model_id, _ in ORION_MODELS]


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


@pytest.mark.integration
class TestChatCompletionsIntegration:
    """Integration tests for chat completions operator.

    These tests call the VLM Run Orion API and require VLMRUN_API_KEY.
    Inspired by VLM Run cookbooks:
    - 12_orion_image_understanding.ipynb
    - 12_orion_video_understanding.ipynb
    """

    @pytest.fixture
    def client(self):
        """Create VLM Run client for testing."""
        from vlmrun.client import VLMRun

        return VLMRun(
            api_key=os.getenv("VLMRUN_API_KEY"),
            base_url="https://agent.vlm.run/v1",
        )

    @requires_api_key
    @pytest.mark.parametrize("model", ["vlmrun-orion-2:fast", "vlmrun-orion-1:fast"])
    def test_image_description(self, client, model):
        """Test basic image description capability across Orion families.

        Runs against both Orion 2 (default) and Orion 1 (still selectable),
        so the backward-compatible path keeps live coverage.

        Cookbook reference: 12_orion_image_understanding.ipynb
        Use case: Generate natural language descriptions of images.
        """
        image_path = TEST_SAMPLES_DIR / "dog.jpg"
        assert image_path.exists(), f"Test image not found: {image_path}"

        # Upload file and get file_id
        uploaded = client.files.upload(file=image_path)

        response = client.agent.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What animal is in this image? Answer in one word."},
                    {"type": "input_file", "file_id": uploaded.id}
                ]
            }],
            temperature=0,
        )

        content = response.choices[0].message.content.lower()
        assert "dog" in content, f"Expected 'dog' in response, got: {content}"

    @requires_api_key
    def test_image_structured_json(self, client):
        """Test structured JSON output with image analysis.

        Cookbook reference: 12_orion_image_understanding.ipynb
        Use case: Extract structured data from images with JSON object format.
        """
        image_path = TEST_SAMPLES_DIR / "dog.jpg"
        assert image_path.exists(), f"Test image not found: {image_path}"

        uploaded = client.files.upload(file=image_path)

        response = client.agent.completions.create(
            model="vlmrun-orion-2:fast",
            messages=[
                {"role": "system", "content": "You are a JSON API. Always respond with valid JSON only, no text."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": 'Analyze this image. Respond with only this JSON format: {"subject": "main subject", "colors": ["color1", "color2"]}'},
                        {"type": "input_file", "file_id": uploaded.id}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        import json
        content = response.choices[0].message.content
        parsed = json.loads(content)
        # JSON object format returns flexible structure based on prompt
        assert isinstance(parsed, dict), f"Expected dict response, got: {type(parsed)}"

    @requires_api_key
    def test_video_summarization(self, client):
        """Test video summarization capability.

        Cookbook reference: 12_orion_video_understanding.ipynb
        Use case: Generate summaries of video content.
        """
        video_path = TEST_SAMPLES_DIR / "sample_video.mp4"
        assert video_path.exists(), f"Test video not found: {video_path}"

        uploaded = client.files.upload(file=video_path)

        response = client.agent.completions.create(
            model="vlmrun-orion-2:fast",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize what happens in this video in one sentence."},
                    {"type": "input_file", "file_id": uploaded.id}
                ]
            }],
            temperature=0,
        )

        content = response.choices[0].message.content
        assert len(content) > 10, f"Expected meaningful summary, got: {content}"

    @requires_api_key
    def test_video_structured_analysis(self, client):
        """Test structured video analysis with JSON output.

        Cookbook reference: 12_orion_video_understanding.ipynb
        Use case: Extract structured information from videos.
        """
        video_path = TEST_SAMPLES_DIR / "sample_video.mp4"
        assert video_path.exists(), f"Test video not found: {video_path}"

        uploaded = client.files.upload(file=video_path)

        response = client.agent.completions.create(
            model="vlmrun-orion-2:fast",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this video. Return JSON with 'description' (what happens) and 'objects' (array of objects seen)."},
                    {"type": "input_file", "file_id": uploaded.id}
                ]
            }],
            response_format={"type": "json_object"},
            temperature=0,
        )

        import json
        content = response.choices[0].message.content
        parsed = json.loads(content)
        # JSON object format returns flexible structure based on prompt
        assert isinstance(parsed, dict), f"Expected dict response, got: {type(parsed)}"

    @requires_api_key
    def test_system_prompt(self, client):
        """Test system prompt functionality.

        Use case: Set context for model responses.
        """
        image_path = TEST_SAMPLES_DIR / "dog.jpg"
        assert image_path.exists(), f"Test image not found: {image_path}"

        uploaded = client.files.upload(file=image_path)

        response = client.agent.completions.create(
            model="vlmrun-orion-2:fast",
            messages=[
                {"role": "system", "content": "You are a veterinarian. Respond professionally."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What breed might this dog be?"},
                        {"type": "input_file", "file_id": uploaded.id}
                    ]
                }
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        assert len(content) > 10, f"Expected meaningful response, got: {content}"

    @requires_api_key
    def test_usage_tracking(self, client):
        """Test that API returns usage/token information.

        Use case: Track API usage for billing and monitoring.
        """
        image_path = TEST_SAMPLES_DIR / "dog.jpg"
        assert image_path.exists(), f"Test image not found: {image_path}"

        uploaded = client.files.upload(file=image_path)

        response = client.agent.completions.create(
            model="vlmrun-orion-2:fast",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "input_file", "file_id": uploaded.id}
                ]
            }],
            temperature=0,
        )

        # Verify usage info is present
        assert hasattr(response, "usage"), "Response should include usage information"
        if response.usage:
            assert hasattr(response.usage, "total_tokens"), "Usage should include total_tokens"

    @requires_api_key
    def test_image_artifact_output(self, client):
        """Test retrieving generated image artifacts.

        Use case: Extract frames from video or generate images and retrieve them.
        Reference: https://docs.vlm.run/agents/api-reference/v1/get-artifact-by-id
        """
        from pydantic import BaseModel, Field
        from vlmrun.types.refs import ImageRef

        video_path = TEST_SAMPLES_DIR / "sample_video.mp4"
        assert video_path.exists(), f"Test video not found: {video_path}"

        uploaded = client.files.upload(file=video_path)

        # Define schema to capture the image artifact ID
        class ExtractedFrameResponse(BaseModel):
            frame_id: ImageRef = Field(..., description="The ID of the extracted frame")

        response = client.agent.completions.create(
            model="vlmrun-orion-2:auto",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the first frame from this video."},
                    {"type": "input_file", "file_id": uploaded.id}
                ]
            }],
            response_format={
                "type": "json_schema",
                "schema": ExtractedFrameResponse.model_json_schema()
            },
            temperature=0,
        )

        # Parse response to get artifact ID
        frame_response = ExtractedFrameResponse.model_validate_json(
            response.choices[0].message.content
        )
        assert frame_response.frame_id.startswith("img_"), f"Expected img_ prefix, got: {frame_response.frame_id}"

        # Retrieve the artifact as PIL Image
        image = client.artifacts.get(
            session_id=response.session_id,
            object_id=frame_response.frame_id,
        )

        # Verify we got a valid image
        import PIL.Image
        assert isinstance(image, PIL.Image.Image), f"Expected PIL Image, got: {type(image)}"
        assert image.size[0] > 0 and image.size[1] > 0, "Image should have valid dimensions"
