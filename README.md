# Adia_TTS Wolof API

## Description

Adia_TTS Wolof API is production ready inference server that provides Text-to-Speech capabilities for the Wolof language. This project solves the accessibility challenges faced by Wolof speakers in digital environments by offering a reliable speech synthesis solution.

The API works with [Adia_TTS](https://huggingface.co/CONCREE/Adia_TTS) model and implements text segmentation strategies to overcome the model's 200-character limitation, making it possible to process texts of any length with natural-sounding results.

## Table of Contents

- [Features](#features)
- [Model Information](#model-information)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [License](#license)
- [Citation](#citation)

## Features

* Wolof speech synthesis with natural voice quality
* Voice style customization through text descriptions
* Automatic text segmentation for long inputs
* Audio concatenation with smooth transitions
* Batch processing for documents
* GPU acceleration support
* FastAPI-based HTTP endpoints


## Installation

### Prerequisites

* Docker
* NVIDIA GPU with CUDA support (strongly recommended)
* Hugging Face API token (Optional)

### Step-by-Step Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/sudoping01/adia-inference-server/

   cd adia-tts-inference-server
   ```

2. Build the Docker image:
   ```bash
   docker build -t adia-tts-wolof .
   ```

3. Run the container with your Hugging Face token:
   ```bash
   docker run --gpus all -p 8080:8080 -e HF_TOKEN=your_hf_token -d adia-tts-wolof
   ```

4. Verify the installation:
   ```bash
   curl http://localhost:8080/health
   ```

## Usage

### Basic Speech Synthesis

Send a POST request to the `/predict` endpoint with Wolof text:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Entreprenariat ci Senegal dafa am solo lool ci yokkuteg koom-koom, di gëna yokk liggéey ak indi gis-gis yu bees ci dëkk bi."
  }' \
  --output speech.wav
```

### Voice Style Customization

You can customize the voice style using the description parameter:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Entreprenariat ci Senegal dafa am solo lool ci yokkuteg koom-koom.",
    "description": "A professional, clear and composed voice, perfect for formal presentations"
  }' \
  --output professional_speech.wav
```

#### Voice Style Examples

| Style | Description Example |
|-------|---------------------|
| Natural | "A warm and natural voice, with a conversational flow" |
| Professional | "A professional, clear and composed voice, perfect for formal presentations" |
| Educational | "A clear and educational voice, with a flow adapted to learning" |

### Advanced Generation Parameters

For more control over the generation process:

```json
{
  "text": "Your Wolof text here",
  "description": "A warm and natural voice, with a conversational flow",
  "temperature": 0.8,
  "max_new_tokens": 1000,
  "do_sample": true,
  "top_k": 50,
  "repetition_penalty": 1.2
}
```

### Batch Processing

Process entire documents using the `/batch_predict` endpoint:

```bash
curl -X POST http://localhost:8080/batch_predict \
  -F "file=@your_document.txt" \
  -F "description=A warm and natural voice" \
  --output document_speech.wav
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Generate speech from text |
| `/batch_predict` | POST | Generate speech from a text file |
| `/health` | GET | Check service health and device status |

## Technical Details

### Text Segmentation

The implementation uses a smart text segmentation approach to overcome the model's 200-character limitation. For each segment, the system:

1. Checks if the text exceeds 200 characters
2. If it does, the system searches backward from the 200-character mark to find the most natural stopping point:
   * First priority: Looks for sentence endings (., !, ?) closest to but before the 200-character limit
   * Second priority: If no sentence ending is found, looks for other punctuation marks (,, ;, :, ...)
   * Last resort: If no punctuation is found, finds the last space before the 200-character mark to avoid cutting words
   * Absolute fallback: If no natural break is found, cuts precisely at 200 characters

This approach ensures that text is segmented at natural pauses whenever possible, which produces more natural-sounding speech when the audio segments are combined. It mimics how a human would naturally pause while reading, resulting in better overall audio quality.

### Audio Processing

For combining audio segments:

* Crossfading (150ms) between segments
* Audio normalization for consistent volume
* Raised cosine functions for smoother transitions
* Processing to reduce artifacts

## Limitations

* The model is limited to 200 characters maximum per inference without segmentation
* Performance may be reduced for very long texts
* Limited handling of numbers and dates
* Initial model loading time is relatively long

## License

This project is available under the MIT license.

