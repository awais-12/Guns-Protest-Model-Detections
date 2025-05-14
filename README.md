# Guns-Protest-Model-Detections
AI-Model Detect The Guns and Protest Model In People 
# Guns-Protest-Model-Detections

This project leverages AI models to detect weapons and protest-related banners in images, videos, and live camera feeds. It uses YOLOv8 models for real-time detection and analysis.

## Features

- Detect weapons and banners in images, videos, and live camera feeds.
- Real-time processing with YOLOv8 models.
- Streamlit-based user interface for easy interaction.

## Prerequisites

- Python 3.10
- Virtual environment support
- Required Python packages (listed in `requirements.txt`)

## Installation and Setup

Follow these steps to set up and run the project:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Guns-Protest-Model-Detectionspython3.10 -m venv venv
   source venv/bin/activate 
 # On Windows: 
    venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run ./cba_ai.py --server.maxUploadSize 1000.

# Directory Structure
├── cba_ai.py                # Main application script
├── data.yaml                # Configuration file
├── models/                  # Pre-trained YOLO models
├── runs/                    # YOLO model outputs
├── test/                    # Test images and labels
├── train/                   # Training data
├── valid/                   # Validation data
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── Test Data/               # Sample test data