"""
Tests for VLM Run FiftyOne plugin operators.
"""

import unittest
from unittest.mock import Mock, patch
import fiftyone as fo
import fiftyone.operators as foo


class TestVMROperators(unittest.TestCase):
    """Test cases for VLM Run operators."""

    def setUp(self):
        """Set up test fixtures."""
        self.dataset = fo.Dataset("test_dataset")
        self.api_key = "test_api_key"
        
        # Create a mock sample
        self.sample = fo.Sample(filepath="/path/to/test.jpg")
        self.sample.media_type = "image"
        self.dataset.add_sample(self.sample)

    def test_vlmrun_object_detection(self):
        """Test object detection operator."""
        from vlmrun import vlmrun_object_detection
        
        with patch('vlmrun.VLMClient') as mock_client:
            mock_instance = Mock()
            mock_instance.run.return_value = {
                "detections": [
                    {"label": "person", "confidence": 0.95, "bbox": [0.1, 0.1, 0.2, 0.3]},
                    {"label": "car", "confidence": 0.87, "bbox": [0.5, 0.6, 0.3, 0.2]}
                ]
            }
            mock_client.return_value = mock_instance
            
            result = vlmrun_object_detection(
                sample_collection=self.dataset,
                api_key=self.api_key,
                grounding=True,
                output_field="detections"
            )
            
            self.assertIsNotNone(result)
            mock_instance.run.assert_called_once_with(
                "image.object-detection",
                "/path/to/test.jpg",
                grounding=True
            )

    def test_vlmrun_person_detection(self):
        """Test person detection operator."""
        from vlmrun import vlmrun_person_detection
        
        with patch('vlmrun.VLMClient') as mock_client:
            mock_instance = Mock()
            mock_instance.run.return_value = {
                "persons": [
                    {"label": "person", "confidence": 0.98, "bbox": [0.1, 0.1, 0.2, 0.3]}
                ]
            }
            mock_client.return_value = mock_instance
            
            result = vlmrun_person_detection(
                sample_collection=self.dataset,
                api_key=self.api_key,
                grounding=True,
                output_field="persons"
            )
            
            self.assertIsNotNone(result)
            mock_instance.run.assert_called_once_with(
                "image.person-detection",
                "/path/to/test.jpg",
                grounding=True
            )

    def test_vlmrun_document_analysis(self):
        """Test document analysis operator."""
        from vlmrun import vlmrun_document_analysis
        
        # Create a document sample
        doc_sample = fo.Sample(filepath="/path/to/test.pdf")
        doc_sample.media_type = "pdf"
        self.dataset.add_sample(doc_sample)
        
        with patch('vlmrun.VLMClient') as mock_client:
            mock_instance = Mock()
            mock_instance.run.return_value = {
                "text_regions": [
                    {"text": "Header", "bbox": [0.1, 0.1, 0.8, 0.1]},
                    {"text": "Body text", "bbox": [0.1, 0.2, 0.8, 0.6]}
                ]
            }
            mock_client.return_value = mock_instance
            
            result = vlmrun_document_analysis(
                sample_collection=self.dataset,
                api_key=self.api_key,
                grounding=True,
                output_field="document_analysis"
            )
            
            self.assertIsNotNone(result)
            mock_instance.run.assert_called_with(
                "document.structure",
                "/path/to/test.pdf",
                grounding=True
            )

    def test_vlmrun_invoice_parsing(self):
        """Test invoice parsing operator."""
        from vlmrun import vlmrun_invoice_parsing
        
        with patch('vlmrun.VLMClient') as mock_client:
            mock_instance = Mock()
            mock_instance.run.return_value = {
                "invoice_data": {
                    "total": "$100.00",
                    "vendor": "Test Company",
                    "date": "2024-01-01"
                }
            }
            mock_client.return_value = mock_instance
            
            result = vlmrun_invoice_parsing(
                sample_collection=self.dataset,
                api_key=self.api_key,
                grounding=True,
                output_field="invoice_data"
            )
            
            self.assertIsNotNone(result)
            mock_instance.run.assert_called_once_with(
                "document.invoice",
                "/path/to/test.jpg",
                grounding=True
            )

    def test_vlmrun_video_transcription(self):
        """Test video transcription operator."""
        from vlmrun import vlmrun_video_transcription
        
        # Create a video sample
        video_sample = fo.Sample(filepath="/path/to/test.mp4")
        video_sample.media_type = "video"
        self.dataset.add_sample(video_sample)
        
        with patch('vlmrun.VLMClient') as mock_client:
            mock_instance = Mock()
            mock_instance.run.return_value = {
                "transcription": "This is a test transcription."
            }
            mock_client.return_value = mock_instance
            
            result = vlmrun_video_transcription(
                sample_collection=self.dataset,
                api_key=self.api_key,
                output_field="transcription"
            )
            
            self.assertIsNotNone(result)
            mock_instance.run.assert_called_with(
                "video.transcription",
                "/path/to/test.mp4"
            )

    def test_vlmrun_video_analysis(self):
        """Test video analysis operator."""
        from vlmrun import vlmrun_video_analysis
        
        # Create a video sample
        video_sample = fo.Sample(filepath="/path/to/test.mp4")
        video_sample.media_type = "video"
        self.dataset.add_sample(video_sample)
        
        with patch('vlmrun.VLMClient') as mock_client:
            mock_instance = Mock()
            mock_instance.run.return_value = {
                "analysis": {
                    "objects": ["person", "car"],
                    "scenes": ["indoor", "outdoor"],
                    "activities": ["walking", "driving"]
                }
            }
            mock_client.return_value = mock_instance
            
            result = vlmrun_video_analysis(
                sample_collection=self.dataset,
                api_key=self.api_key,
                analysis_type="comprehensive",
                output_field="video_analysis"
            )
            
            self.assertIsNotNone(result)
            mock_instance.run.assert_called_with(
                "video.comprehensive",
                "/path/to/test.mp4"
            )

    def test_error_handling(self):
        """Test error handling in operators."""
        from vlmrun import vlmrun_object_detection
        
        with patch('vlmrun.VLMClient') as mock_client:
            mock_instance = Mock()
            mock_instance.run.side_effect = Exception("API Error")
            mock_client.return_value = mock_instance
            
            # Should not raise an exception
            result = vlmrun_object_detection(
                sample_collection=self.dataset,
                api_key=self.api_key,
                grounding=True,
                output_field="detections"
            )
            
            self.assertIsNotNone(result)

    def test_unsupported_media_type(self):
        """Test handling of unsupported media types."""
        from vlmrun import vlmrun_object_detection
        
        # Create a sample with unsupported media type
        unsupported_sample = fo.Sample(filepath="/path/to/test.txt")
        unsupported_sample.media_type = "text"
        self.dataset.add_sample(unsupported_sample)
        
        with patch('vlmrun.VLMClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            result = vlmrun_object_detection(
                sample_collection=self.dataset,
                api_key=self.api_key,
                grounding=True,
                output_field="detections"
            )
            
            # Should only process the image sample, not the text sample
            self.assertIsNotNone(result)
            mock_instance.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
