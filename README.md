# Smart Mirror

AI-powered smart mirror with computer vision and VLM reasoning capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    JETSON ORIN NANO (Edge CV)                       │
├─────────────────────────────────────────────────────────────────────┤
│  Face Detection │ Pose Estimation │ Gesture Recognition │ Health   │
│       ↓                ↓                  ↓                 ↓       │
│                    Event Router + Frame Sampler                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ JSON + Selected Frames
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    PC w/ RTX 5080 (VLM Reasoning)                     │
│  Ollama API → VLM (Qwen2.5-VL) → Reasoning + Recommendations         │
└───────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
smart-mirror/
├── edge/                    # Jetson Orin Nano code
│   ├── cv/                  # Computer vision modules
│   │   ├── detectors/       # Face detection
│   │   ├── estimators/      # Pose & gesture
│   │   ├── health/          # rPPG heart rate
│   │   └── skin/            # Lesion classification
│   ├── capture/             # Camera handling
│   ├── router/              # Event routing to VLM
│   └── main.py              # Edge entry point
├── server/                  # PC server code
│   ├── vlm/                 # Ollama VLM client
│   ├── api/                 # REST/WebSocket API
│   └── main.py              # Server entry point
├── shared/                  # Shared code
│   ├── protocol/            # Communication messages
│   └── schemas/             # VLM request/response schemas
├── config/                  # Configuration
└── models/                  # Trained models
    └── skin_cancer/         # LSTM model from prototype
```

## Setup

### PC Server (VLM)

1. Install Ollama: https://ollama.ai
2. Pull the vision model:
   ```bash
   ollama pull qwen2.5-vl:7b
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements-server.txt
   ```
4. Run server:
   ```bash
   python -m server.main
   ```

### Jetson Orin Nano (Edge)

1. Install NVIDIA JetPack SDK
2. Install dependencies:
   ```bash
   pip install -r requirements-edge.txt
   ```
3. Run edge application:
   ```bash
   python -m edge.main
   ```

## CV Modules

| Module | Model | Purpose |
|--------|-------|---------|
| Face Detection | YOLOv8n-face | Detect and track faces |
| Pose Estimation | MoveNet | Body pose keypoints |
| Gesture Recognition | MediaPipe Hands | Hand gesture commands |
| Heart Rate | rPPG (CHROM) | Non-contact heart rate |
| Skin Analysis | MobileNet + LSTM | Lesion classification |

## VLM Integration

The VLM (Qwen2.5-VL) handles:
- Visual understanding of the user
- Natural language conversation
- Health recommendations
- Skin analysis interpretation
- Gesture command responses

## Configuration

Edit `config/config.json` or use environment-specific configs:
- `get_jetson_config()` - Optimized for Jetson
- `get_development_config()` - For local development

## Hardware Requirements

### Edge Device (Jetson Orin Nano)
- 4-8GB RAM
- Camera module (USB or CSI)
- Network connection to PC

### VLM Server (PC)
- 32GB RAM
- RTX 5080 GPU
- Ollama installed

### Mirror Hardware
- Two-way mirror acrylic sheet
- Touchscreen display
- Frame/enclosure
