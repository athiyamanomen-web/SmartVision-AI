# SmartVision AI
## Intelligent Multi-Class Object Recognition System

SmartVision AI is an end-to-end computer vision application for multi-class image classification and multi-object detection.

The project combines four transfer-learning CNN architectures with a fine-tuned YOLOv8 detector and provides an interactive Streamlit application for inference, comparison, visualization, and live webcam detection.

---

## Live Application

- **Live Demo:** [SmartVision AI](https://smartvision-ai-i93kgzhvpxx33hlftoqtnp.streamlit.app/)
- **GitHub Repository:** [Source Code](https://github.com/athiyamanomen-web/SmartVision-AI)
- **Deployment Platform:** Streamlit Community Cloud
- **Application Status:** Deployed and functionally tested

The application supports desktop and mobile browsers.
No local installation is required to access the live demo.

---

## Features

- 25-class image classification
- Multi-object detection using YOLOv8
- Four CNN classification architectures
- Top-5 classification predictions
- Confidence scores and bounding-box visualization
- Optional ResNet50 verification of detected objects
- Model performance comparison
- Confusion matrix visualization
- Class-wise detection performance
- Adjustable confidence and NMS thresholds
- Live webcam object detection
- GPU acceleration with CPU fallback
- Streamlit-based interactive interface
- JPEG and PNG file signature validation
- Image integrity verification
- Graceful image decoding error handling
- Browser-based WebRTC streaming
- STUN and TURN integration
- Secure cloud credential configuration
- Desktop and mobile browser compatibility

---

## Supported Classes

SmartVision AI supports 25 object classes:

person, bicycle, car, motorcycle, airplane, bus, train, truck, traffic light, stop sign, bench, bird, cat, dog, horse, cow, elephant, bottle, cup, bowl, pizza, cake, chair, potted plant, bed.

---

## Dataset

The project uses a curated subset of the COCO dataset.

### Classification Dataset

- 25 classes
- 100 images per class
- 2,500 classification images
- Image size: 224 x 224 RGB
- Split: 70% training / 15% validation / 15% testing
- Source-group leakage-free splitting

### Object Detection Dataset

- 1,817 unique images
- 2,500 class-image assignments
- 10,988 annotated objects
- 25 classes
- YOLO-format bounding-box annotations
- Leakage-free train/validation/test split

The optimized YOLO training dataset was later expanded during fine-tuning to improve weaker detection classes.

---

## Classification Models

Four ImageNet-pretrained transfer-learning architectures were trained and evaluated:

- VGG16
- ResNet50
- MobileNetV2
- EfficientNetB0

### Final Test Performance

| Model | Accuracy | Precision | Recall | F1 | Top-5 | Avg Inference | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 86.13% | 88.19% | 86.13% | 86.27% | 97.07% | 11.27 ms | 88.73 |
| EfficientNetB0 | 84.27% | 85.65% | 84.27% | 84.16% | 97.60% | 15.25 ms | 65.56 |
| VGG16 | 81.60% | 83.12% | 81.60% | 81.39% | 97.87% | 4.97 ms | 201.17 |
| MobileNetV2 | 79.73% | 81.48% | 79.73% | 79.69% | 95.47% | 10.45 ms | 95.73 |

**Benchmark note:** FPS values are derived from average
model inference latency. They are not measured live
webcam frame rates.

### Selected Classification Model

**ResNet50** is used as the primary classifier because it achieved the highest final test accuracy and F1 score.

---

## Object Detection

SmartVision AI uses a fine-tuned **YOLOv8s** detector.

### Final YOLOv8 Test Performance

| Metric | Result |
|---|---:|
| Test images | 356 |
| Test instances | 1,714 |
| Precision | 86.74% |
| Recall | 71.66% |
| mAP@0.5 | 80.78% |
| mAP@0.5:0.95 | 64.32% |

The optimized detector improved mAP@0.5 from **77.85% to 80.78%**.

The final detector exceeded the project's formal grading requirement of mAP@0.5 greater than 75%.

---

## End-to-End Pipeline

1. User uploads an image or supplies a webcam frame.
2. YOLOv8 detects supported objects.
3. Non-Maximum Suppression removes overlapping detections.
4. Confidence filtering removes low-confidence predictions.
5. Bounding boxes are clipped to valid image coordinates.
6. Objects are displayed with labels and confidence scores.
7. Optional ResNet50 verification classifies detected crops.
8. Results are visualized through Streamlit.

---

## Streamlit Application

The application contains six pages:

### Home
Project overview, features, dataset summary, instructions, and sample prediction demonstrations.

### Image Classification
Upload an image, run any of the four CNN models, compare models, and display Top-5 predictions.

### Object Detection
Run YOLOv8 multi-object detection with adjustable confidence and NMS thresholds and optional ResNet50 verification.

### Model Performance
View classification comparisons, confusion matrices, training curves, YOLO metrics, class-wise performance, and failure cases.

### Live Webcam
Browser-based real-time YOLO detection with bounding boxes, confidence scores, FPS, and latency.

### About
Project documentation, dataset information, architectures, technical stack, system pipeline, and developer information.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch, torchvision |
| Classification | VGG16, ResNet50, MobileNetV2, EfficientNetB0 |
| Object Detection | Ultralytics YOLOv8s |
| Computer Vision | OpenCV, Pillow |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Streamlit |
| Web Application | Streamlit |
| Live Video | streamlit-webrtc, PyAV, WebRTC |
| Network Connectivity | STUN, Metered TURN |
| Dataset Source | COCO via Hugging Face |
| Version Control | Git, GitHub, Git LFS |
| Cloud Deployment | Streamlit Community Cloud |
| Container Configuration | Docker |

---

## Project Structure

```text
SmartVision-AI/
|-- models/
|-- notebooks/
|-- results/
|-- runs/
|   `-- detection/
|       `-- yolov8s_v2_targeted_15ep/
|           `-- weights/
|               `-- best.pt
|-- streamlit_app/
|   |-- app.py
|   |-- pages/
|   `-- utils/
|-- .gitattributes
|-- .gitignore
|-- requirements.txt
|-- packages.txt
|-- Dockerfile
`-- README.md
```

The dataset is excluded from Git because of its size. Dataset preparation and reproducibility steps are documented in the notebooks.

---

## Local Installation

### Prerequisites

- Python 3.11 recommended
- Git and Git LFS
- Internet connection for initial downloads
- NVIDIA CUDA-compatible GPU (optional)

### 1. Clone the repository

```bash
git lfs install
git clone https://github.com/athiyamanomen-web/SmartVision-AI.git
cd SmartVision-AI
git lfs pull
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux users may also need the system dependencies
listed in packages.txt.

For GPU acceleration, install a compatible
CUDA-enabled PyTorch build when necessary.

### 5. Run the application

```bash
streamlit run streamlit_app/app.py
```

Open the local URL displayed in the terminal.

The application uses a compatible GPU when available
and otherwise falls back to CPU inference.

---

## Model Files

Model checkpoints are managed using **Git LFS**.

```bash
git lfs install
```

Final deployment models:

- vgg16_final_best.pth
- resnet50_final_best.pth
- mobilenetv2_final_best.pth
- efficientnetb0_final_best.pth
- YOLOv8s best.pt

---

## Deployment

### Production Deployment

SmartVision AI is deployed on Streamlit Community Cloud.

**Live Application:** [https://smartvision-ai-i93kgzhvpxx33hlftoqtnp.streamlit.app/](https://smartvision-ai-i93kgzhvpxx33hlftoqtnp.streamlit.app/)

| Configuration | Value |
|---|---|
| Hosting Platform | Streamlit Community Cloud |
| Source Repository | GitHub |
| Python Version | 3.11 |
| Application Entry Point | streamlit_app/app.py |
| Model Storage | Git LFS |
| Cloud Inference Device | CPU |
| Local Inference | GPU with CPU fallback |
| Python Dependencies | requirements.txt |
| Linux Dependencies | packages.txt |
| Webcam Connectivity | WebRTC with STUN and TURN |

### Deployment Platform Selection

The original project specification requested deployment
on Hugging Face Spaces using the native Streamlit SDK.

During deployment, the native Streamlit SDK was no longer
offered, and Docker-based deployment was presented as
a paid option on the developer's account.

Streamlit Community Cloud was selected as an alternative
to preserve the existing application architecture and
avoid additional hosting costs.

A Dockerfile is retained for potential future
container-based deployment. Docker deployment has
not been validated.

### Cloud Webcam Connectivity

The live webcam feature uses streamlit-webrtc to transmit
browser video frames to the inference server.

A STUN server assists direct connection establishment.
Metered TURN provides relay connectivity when a direct
WebRTC connection cannot be established.

TURN integration resolved the observed cloud webcam
connection timeout during functional testing.

TURN credentials are configured through Streamlit Cloud
Secrets and must not be committed to GitHub.

Example configuration:

```toml
[turn]
urls = [
    "turn:global.relay.metered.ca:80",
    "turn:global.relay.metered.ca:80?transport=tcp",
    "turn:global.relay.metered.ca:443",
    "turns:global.relay.metered.ca:443?transport=tcp"
]
username = "YOUR_TURN_USERNAME"
credential = "YOUR_TURN_PASSWORD"
```

The example contains placeholders only.
Valid credentials are required for TURN connectivity.

Local webcam operation was also tested successfully
after the WebRTC configuration update.

---

## Production Testing

| Test | Result |
|---|---|
| Home page | Passed |
| Four CNN classification models | Passed |
| YOLOv8s multi-object detection | Passed |
| Optional ResNet50 verification | Passed |
| Model performance dashboard | Passed |
| Local webcam detection | Passed |
| Cloud webcam detection | Passed |
| About page | Passed |
| Google Chrome compatibility | Passed |
| Microsoft Edge compatibility | Passed |
| Mobile browser compatibility | Passed |
| Valid image upload | Passed |
| Invalid image handling | Passed |

These results represent functional testing, not
exhaustive load, security or stress testing.

### Error Handling

The application validates JPEG and PNG file signatures,
checks image integrity and handles decoding failures.

Invalid or corrupted uploads are rejected with a
user-friendly error message.

Model loading and inference errors are handled
through Streamlit feedback messages.

---

## Notes and Limitations

### Dataset and Classification

- Models were trained on a curated 25-class COCO subset.
- Classification models were trained primarily on
  extracted single-object crops.
- Full-scene images containing multiple objects may
  produce ambiguous or incorrect classification results.
- Unsupported objects may be incorrectly assigned
  to one of the supported classes.

### Performance Targets

| Model | Expected Accuracy / mAP | Achieved |
|---|---:|---:|
| VGG16 | 80-85% | 81.60% |
| ResNet50 | 85-90% | 86.13% |
| MobileNetV2 | 82-87% | 79.73% |
| EfficientNetB0 | 88-93% | 84.27% |
| YOLOv8s mAP@0.5 | 85-90% | 80.78% |

VGG16 and ResNet50 achieved their expected ranges.
MobileNetV2 and EfficientNetB0 remained below their
expected accuracy targets.

YOLOv8s exceeded the formal evaluation threshold of
75% mAP@0.5 but remained below the expected range.

### Evaluation Methodology

- Reported metrics are specific to the project dataset.
- YOLO optimization was informed by earlier weak-class
  evaluation results.
- The V1 result of 77.85% mAP@0.5 is retained as an
  earlier performance baseline.
- The final V2 result is not a completely independent
  test-set estimate because earlier test findings
  informed optimization.

### Inference and Deployment

- Inference speed depends on hardware and configuration.
- Benchmark FPS values are derived from model latency,
  not measured end-to-end webcam frame rates.
- Cloud inference uses CPU resources.
- Cloud cold starts may increase loading time.
- Dynamic quantization was evaluated, but FP32 was
  retained because quantization did not improve latency.
- Webcam performance depends on browser permissions,
  network conditions and server resources.
- Metered TURN is subject to its free bandwidth quota
  and service availability.
- Streamlit Community Cloud was used instead of the
  originally specified Hugging Face Spaces platform.
- Acceptance of the alternative platform is subject
  to project evaluator approval.

---

## Future Improvements

- Larger YOLO variants
- Higher-resolution detector training
- Additional data for weaker classes
- Model pruning and quantization
- ONNX or TensorRT inference
- Additional object categories
- Improved mobile deployment
- Real-time object tracking

---

## Author

Developed as part of the **SmartVision AI - Intelligent Multi-Class Object Recognition System** project.
