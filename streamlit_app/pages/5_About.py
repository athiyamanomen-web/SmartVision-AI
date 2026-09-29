
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="About | SmartVision AI",
    page_icon="ℹ️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("ℹ️ About SmartVision AI")

st.subheader(
    "Intelligent Multi-Class Object Recognition System"
)

st.write(
    """
    SmartVision AI is a deep-learning computer vision project
    designed for image classification and multi-object detection.

    The system combines four transfer-learning CNN architectures
    with a fine-tuned YOLOv8s object detector to recognize
    25 selected COCO object classes.
    """
)

st.divider()


# ============================================================
# PROJECT DOCUMENTATION
# ============================================================

st.header("📘 Project Overview")

st.write(
    """
    SmartVision AI demonstrates an end-to-end computer vision
    workflow covering dataset preparation, exploratory analysis,
    transfer learning, object detection, model evaluation,
    inference-pipeline development and interactive deployment.

    The application provides two primary prediction workflows:

    **Image Classification**
    - Classifies an uploaded image using trained CNN models.
    - Supports VGG16, ResNet50, MobileNetV2 and EfficientNetB0.
    - Displays Top-5 predictions and confidence scores.
    - Allows direct comparison of all four models.

    **Object Detection**
    - Detects multiple objects using the fine-tuned YOLOv8s model.
    - Displays bounding boxes, class labels and confidence scores.
    - Supports adjustable confidence and NMS IoU thresholds.
    - Provides optional ResNet50 classification verification.
    """
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.header("📚 Dataset")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Object Classes",
    "25"
)

col2.metric(
    "Classification Images",
    "2,500"
)

col3.metric(
    "Detection Images",
    "1,817"
)

st.write(
    """
    The dataset was created from a curated subset of COCO.

    **Classification dataset**
    - 25 classes
    - 100 images per class
    - 2,500 object crops
    - 224 × 224 RGB images
    - 70% training / 15% validation / 15% testing

    **Detection dataset**
    - 1,817 unique source images
    - 10,988 annotated objects
    - YOLO-format bounding-box annotations
    - Leakage-free train, validation and test splits
    """
)


# ============================================================
# SUPPORTED CLASSES
# ============================================================

st.subheader("Supported Object Classes")

classes = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "traffic light", "stop sign",
    "bench", "bird", "cat", "dog", "horse",
    "cow", "elephant", "bottle", "cup", "bowl",
    "pizza", "cake", "chair", "potted plant", "bed"
]

st.write(
    ", ".join(classes)
)


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

st.header("🧠 Model Architectures")

model_col1, model_col2 = st.columns(2)


with model_col1:

    st.subheader("Classification Models")

    st.markdown(
        """
        **VGG16**
        - ImageNet pretrained
        - Frozen convolutional backbone
        - Custom fully connected classifier

        **ResNet50**
        - ImageNet pretrained
        - Partial fine-tuning
        - Custom classification head
        - Selected primary classifier

        **MobileNetV2**
        - Lightweight ImageNet pretrained architecture
        - Designed for efficient inference

        **EfficientNetB0**
        - ImageNet pretrained
        - Partial fine-tuning
        - Efficient compound-scaled architecture
        """
    )


with model_col2:

    st.subheader("Object Detection Model")

    st.markdown(
        """
        **YOLOv8s**

        - Fine-tuned for the 25 SmartVision classes
        - Multi-object detection
        - Bounding-box localization
        - Confidence scoring
        - Non-Maximum Suppression
        - GPU acceleration supported
        """
    )


# ============================================================
# FINAL MODEL RESULTS
# ============================================================

st.header("📊 Final Model Results")

result_col1, result_col2 = st.columns(2)


with result_col1:

    st.subheader("ResNet50 Classification")

    st.metric(
        "Test Accuracy",
        "86.13%"
    )

    st.metric(
        "F1 Score",
        "86.27%"
    )

    st.metric(
        "Top-5 Accuracy",
        "97.07%"
    )


with result_col2:

    st.subheader("YOLOv8s Detection")

    st.metric(
        "Precision",
        "86.74%"
    )

    st.metric(
        "Recall",
        "71.66%"
    )

    st.metric(
        "mAP@0.5",
        "80.78%"
    )

    st.metric(
        "mAP@0.5:0.95",
        "64.32%"
    )


# ============================================================
# TECHNICAL STACK
# ============================================================

st.header("🛠️ Technical Stack")

stack_col1, stack_col2, stack_col3 = st.columns(3)


with stack_col1:

    st.markdown(
        """
        **Deep Learning**
        - PyTorch
        - Torchvision
        - Ultralytics YOLOv8
        - Transfer Learning
        """
    )


with stack_col2:

    st.markdown(
        """
        **Data & Vision**
        - COCO
        - Hugging Face Datasets
        - PIL
        - OpenCV
        - NumPy
        - Pandas
        """
    )


with stack_col3:

    st.markdown(
        """
        **Application**
        - Streamlit
        - Python
        - Matplotlib
        - CUDA acceleration
        """
    )


# ============================================================
# SYSTEM PIPELINE
# ============================================================

st.header("⚙️ System Pipeline")

st.markdown(
    """
    **Classification**

    Image Upload → Preprocessing → CNN Model →
    Softmax Probabilities → Top-5 Predictions

    **Object Detection**

    Image Upload → YOLOv8s → Confidence Filtering →
    NMS → Bounding Boxes → Optional ResNet50 Verification →
    Visualization
    """
)


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

st.header("⚡ Performance Optimization")

st.markdown(
    """
    SmartVision AI includes:

    - GPU-aware inference with CPU fallback
    - `torch.inference_mode()` for inference
    - Batch-processing support
    - Cloud-oriented memory management
    - Optional CNN verification to reduce unnecessary computation
    - Dynamic INT8 quantization experiment

    Dynamic INT8 quantization preserved the tested ResNet50
    prediction but did not improve CPU latency on the development
    machine. Therefore, FP32 ResNet50 was retained as the primary
    classifier.
    """
)


# ============================================================
# DEVELOPER
# ============================================================

st.header("👨‍💻 Developer Information")

st.write(
    """
    **Project:** SmartVision AI  
    **Project Type:** Deep Learning / Computer Vision  
    **Application:** Intelligent Multi-Class Object Recognition System
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SmartVision AI | Intelligent Multi-Class Object Recognition System"
)
