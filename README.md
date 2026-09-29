# SmartVision AI
## Intelligent Multi-Class Object Recognition System

SmartVision AI is an end-to-end computer vision application for multi-class image classification and multi-object detection.

The project combines four transfer-learning CNN architectures with a fine-tuned YOLOv8 detector and provides an interactive Streamlit application for inference, comparison, visualization, and live webcam detection.

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

- Python
- PyTorch
- torchvision
- Ultralytics YOLOv8
- OpenCV
- Pillow
- NumPy
- Pandas
- Matplotlib
- Streamlit
- streamlit-webrtc
- PyAV
- Git and Git LFS
- Docker
- Hugging Face Spaces

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
|-- Dockerfile
`-- README.md
```

The dataset is excluded from Git because of its size. Dataset preparation and reproducibility steps are documented in the notebooks.

---

## Local Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SmartVision-AI
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
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run streamlit_app/app.py
```

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

The application is prepared for deployment on Hugging Face Spaces using Streamlit in a Docker-based Space.

The application uses deployment-safe relative paths and supports GPU execution when available with CPU fallback.

---

## Notes and Limitations

- Results are based on the curated COCO subset used in this project.
- Inference speed depends on hardware and runtime configuration.
- Webcam performance depends on browser permissions and deployment hardware.
- V2 YOLO optimization was informed by weak-class analysis from an earlier evaluation; the V1 test result is retained as the untouched benchmark.

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
