
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO


# ============================================================
# PATH SETUP
# ============================================================

APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from utils.model_loader import (
    get_device,
    load_yolo,
    load_resnet50
)

from utils.inference import (
    detect_objects,
    verify_detections,
    draw_detections
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Object Detection | SmartVision AI",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# CACHED MODEL LOADERS
# ============================================================

@st.cache_resource
def get_yolo_model():
    return load_yolo()


@st.cache_resource
def get_resnet_model():

    device = get_device(
        prefer_gpu=True
    )

    model = load_resnet50(
        device
    )

    return model, device


# ============================================================
# HEADER
# ============================================================

st.title("🎯 Object Detection")

st.write(
    """
    Upload an image and detect multiple objects using the
    fine-tuned SmartVision YOLOv8s detector.
    """
)

st.info(
    "YOLOv8s supports the 25 selected SmartVision COCO classes."
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header(
    "Detection Settings"
)

confidence_threshold = (
    st.sidebar.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05
    )
)

nms_iou_threshold = (
    st.sidebar.slider(
        "NMS IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.70,
        step=0.05
    )
)

verify_with_resnet = (
    st.sidebar.checkbox(
        "Verify detections with ResNet50",
        value=False
    )
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is None:

    st.markdown(
        """
        ### How to use

        1. Upload a JPG, JPEG or PNG image.
        2. Adjust the confidence threshold if required.
        3. Adjust the NMS IoU threshold.
        4. Optionally enable ResNet50 verification.
        5. SmartVision AI will display detected objects with
           bounding boxes, labels and confidence scores.
        """
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:
    image_bytes = uploaded_file.getvalue()

    # Validate supported image signatures
    is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    if not (is_jpeg or is_png):
        raise ValueError("Unsupported or invalid image format")

    # Verify image integrity
    with Image.open(BytesIO(image_bytes)) as validation_image:
        if validation_image.format not in ("JPEG", "PNG"):
            raise ValueError("Unsupported image format")

        validation_image.verify()

    # Reopen and fully decode the verified image
    with Image.open(BytesIO(image_bytes)) as decoded_image:
        image = decoded_image.convert("RGB")
        image.load()

except (
    OSError,
    ValueError,
    ImportError,
    SyntaxError,
    Image.DecompressionBombError
):
    st.error(
        "Invalid or corrupted image. "
        "Please upload a valid JPG or PNG."
    )
    st.stop()


# ============================================================
# LOAD YOLO
# ============================================================

try:
    with st.spinner(
        "Loading SmartVision YOLOv8s..."
    ):

        yolo_model = get_yolo_model()
except Exception:
    st.error('Detection or verification failed. Check model files and available memory, then retry.')
    st.stop()


# ============================================================
# SELECT INFERENCE DEVICE
# ============================================================

device = get_device(
    prefer_gpu=True
)

yolo_device = (
    0
    if device.type == "cuda"
    else "cpu"
)


# ============================================================
# RUN DETECTION
# ============================================================

try:
    with st.spinner(
        "Detecting objects..."
    ):

        start_time = time.perf_counter()

        detections = detect_objects(
            yolo_model=yolo_model,
            image=image,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=yolo_device
        )

        detection_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000
except Exception:
    st.error('Detection or verification failed. Check model files and available memory, then retry.')
    st.stop()


# ============================================================
# OPTIONAL CNN VERIFICATION
# ============================================================

try:
    if (
        verify_with_resnet
        and detections
    ):

        with st.spinner(
            "Verifying detections with ResNet50..."
        ):

            resnet_model, resnet_device = (
                get_resnet_model()
            )

            detections = verify_detections(
                image=image,
                detections=detections,
                resnet_model=resnet_model,
                device=resnet_device
            )
except Exception:
    st.error('Detection or verification failed. Check model files and available memory, then retry.')
    st.stop()


# ============================================================
# DRAW RESULTS
# ============================================================

result_image = draw_detections(
    image=image,
    detections=detections
)


# ============================================================
# MAIN RESULTS
# ============================================================

left_col, right_col = st.columns(
    [1.5, 1]
)


with left_col:

    st.subheader(
        "Detection Result"
    )

    st.image(
        result_image,
        use_container_width=True
    )


with right_col:

    st.subheader(
        "Detection Summary"
    )

    metric1, metric2 = st.columns(2)

    metric1.metric(
        "Objects Detected",
        len(detections)
    )

    metric2.metric(
        "Processing Time",
        f"{detection_time_ms:.2f} ms"
    )

    st.write(
        "**Inference device:**",
        str(device).upper()
    )

    st.write(
        "**Confidence threshold:**",
        f"{confidence_threshold:.2f}"
    )

    st.write(
        "**NMS IoU threshold:**",
        f"{nms_iou_threshold:.2f}"
    )

    st.write(
        "**ResNet50 verification:**",
        (
            "Enabled"
            if verify_with_resnet
            else "Disabled"
        )
    )


# ============================================================
# DETECTION TABLE
# ============================================================

st.divider()

st.subheader(
    "Detected Objects"
)


if not detections:

    st.warning(
        "No objects were detected above the selected "
        "confidence threshold."
    )

else:

    rows = []

    for index, detection in enumerate(
        detections,
        start=1
    ):

        row = {
            "#": index,
            "Class":
                detection["class"],

            "YOLO Confidence (%)":
                detection["confidence"]
                * 100,

            "Bounding Box":
                str(detection["bbox"])
        }

        if "cnn_class" in detection:

            row[
                "ResNet50 Prediction"
            ] = detection[
                "cnn_class"
            ]

            row[
                "ResNet50 Confidence (%)"
            ] = (
                detection[
                    "cnn_confidence"
                ]
                * 100
            )

            row[
                "Class Match"
            ] = (
                "Yes"
                if detection[
                    "class_match"
                ]
                else "No"
            )

        rows.append(row)


    detection_df = pd.DataFrame(
        rows
    )


    percentage_columns = [
        "YOLO Confidence (%)",
        "ResNet50 Confidence (%)"
    ]

    for column in percentage_columns:

        if column in detection_df.columns:

            detection_df[column] = (
                detection_df[column]
                .round(2)
            )


    st.dataframe(
        detection_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PIPELINE INFORMATION
# ============================================================

with st.expander(
    "How detection works"
):

    st.markdown(
        """
        **SmartVision detection pipeline**

        Uploaded Image  
        ↓  
        YOLOv8s Object Detection  
        ↓  
        Confidence Filtering  
        ↓  
        Non-Maximum Suppression (NMS)  
        ↓  
        Bounding Box Refinement  
        ↓  
        Optional ResNet50 Verification  
        ↓  
        Final Visualization
        """
    )


st.divider()

st.caption(
    "SmartVision AI | "
    "YOLOv8s Multi-Object Detection"
)
