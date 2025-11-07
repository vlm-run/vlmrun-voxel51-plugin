# VLM Run Plugin

A plugin that provides operators for extracting structured data from visual
sources using [VLM Run](https://vlm.run)'s vision-language models.

![VLM Run Plugin Demo](gifs/plugin_overview.gif)

## Installation

```shell
fiftyone plugins download \
    https://github.com/vlm-run/vlmrun-voxel51-plugin
```

Install the required dependencies:

```shell
fiftyone plugins requirements @vlm-run/vlmrun-voxel51-plugin --install
```

Refer to the [FiftyOne Plugins documentation](https://docs.voxel51.com/plugins/index.html) for
more information about managing downloaded plugins and developing plugins
locally.

## Configuration

Set your VLM Run API key as an environment variable:

```shell
export VLM_API_KEY="your-api-key-here"
```

You can obtain an API key from [vlm.run](https://app.vlm.run/dashboard/settings/api-keys).

Alternatively, you can provide the API key directly when running operators in
the FiftyOne App.

## Usage

1.  Launch the App:

```py
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart", max_samples=10)
session = fo.launch_app(dataset)
```

2.  Press `` ` `` or click the `Browse operations` action to open the Operators
    list

3.  Select any of the operators listed below!

## Operators

### object_detection

Detect and localize common objects in images with bounding box coordinates.

This operator uses VLM Run's object detection domain with visual grounding to
extract precise bounding boxes in normalized xywh format:

```py
# Detect objects with visual grounding
from vlmrun import VLMRun

client = VLMRun(api_key="your-api-key")
result = client.run(
    "image.object-detection",
    image_path,
    grounding=True
)
```

The operator adds detections to your dataset with:
- Bounding boxes in normalized coordinates
- Confidence scores for each detection
- Object labels

### person_detection

Specialized person detection with enhanced accuracy for human-centric
applications.

This operator uses VLM Run's person detection domain optimized for detecting
people in various scenarios:

```py
# Detect people with high accuracy
client.run(
    "image.person-detection",
    image_path,
    grounding=True
)
```

Features:
- High-accuracy person detection
- Optimized for challenging scenarios (crowds, occlusions)
- Precise bounding boxes with confidence scores

### document_analysis

Extract text and analyze document structure from PDFs and images.

This operator leverages VLM Run's document analysis capabilities:

```py
# Analyze document structure and extract text
client.run(
    "document.analysis",
    document_path,
    grounding=True
)
```

The operator extracts:
- Text content with spatial coordinates
- Document structure (headers, paragraphs, sections)
- Tables and figures with bounding boxes
- Reading order information

### invoice_parsing

Extract structured data from invoice documents with field-level visual
grounding.

This operator uses VLM Run's invoice parsing domain:

```py
# Parse invoice and extract structured data
client.run(
    "document.invoice-parsing",
    invoice_path,
    grounding=True
)
```

Extracts:
- Invoice totals and line items
- Vendor and customer information
- Dates, invoice numbers, and payment terms
- Tax and discount information
- Visual grounding for each extracted field (optional)

### layout_detection

Analyze document layout and identify structural elements with precise
localization.

This operator uses VLM Run's layout detection capabilities:

```py
# Detect layout elements in documents
client.run(
    "document.layout-detection",
    document_path,
    grounding=True
)
```

Identifies:
- Text regions and columns
- Headers, footers, and body text
- Tables, figures, and images
- Captions and footnotes
- Bounding boxes for each layout element

![Layout Detection Example](img/layout_detection.jpg)

### video_transcription

Transcribe audio and analyze video content with multiple analysis modes.

This operator provides comprehensive video analysis using VLM Run's video
understanding capabilities:

```py
# Transcribe video with various analysis modes
client.run(
    "video.transcription",  # or other modes
    video_path
)
```

Supported modes:
- **transcription**: Audio-to-text transcription with timestamps
- **comprehensive**: Full video analysis (audio + visual + activities)
- **objects**: Object detection across video frames
- **scenes**: Scene classification and changes
- **activities**: Activity and action recognition

Each mode provides temporal information and can be combined for comprehensive
video understanding.

## Visual Grounding

When enabled, visual grounding provides bounding box coordinates in normalized
xywh format:

- `x`: horizontal position of top-left corner (0-1)
- `y`: vertical position of top-left corner (0-1)
- `w`: width of the bounding box (0-1)
- `h`: height of the bounding box (0-1)

This allows for precise localization of detected objects, text regions, or
document elements directly on your images.

## Supported Formats

- **Images**: JPEG, PNG, BMP, TIFF, and other common formats
- **Documents**: PDF files and document images
- **Videos**: MP4, AVI, MOV, MKV, WEBM, FLV, WMV, M4V

## Execution Modes

All operators support two execution modes:

- **Immediate**: Process immediately in the FiftyOne App (default)
- **Delegated**: Queue for background processing (requires
  [orchestrator setup](https://docs.voxel51.com/plugins/using_plugins.html#delegating-plugin-operations))

## Learn More

- [VLM Run Documentation](https://docs.vlm.run)
- [FiftyOne Documentation](https://docs.voxel51.com)
- [Plugin Development Guide](https://docs.voxel51.com/plugins/index.html)
