
import sys
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
    load_vgg16,
    load_resnet50,
    load_mobilenetv2,
    load_efficientnetb0
)

from utils.inference import classify_image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Image Classification | SmartVision AI",
    page_icon="🖼️",
    layout="wide"
)


# ============================================================
# CACHED MODEL LOADERS
# ============================================================

@st.cache_resource
def get_vgg16():
    device = get_device(prefer_gpu=True)
    return load_vgg16(device), device


@st.cache_resource
def get_resnet50():
    device = get_device(prefer_gpu=True)
    return load_resnet50(device), device


@st.cache_resource
def get_mobilenetv2():
    device = get_device(prefer_gpu=True)
    return load_mobilenetv2(device), device


@st.cache_resource
def get_efficientnetb0():
    device = get_device(prefer_gpu=True)
    return load_efficientnetb0(device), device


# ============================================================
# HEADER
# ============================================================

st.title("🖼️ Image Classification")

st.write(
    """
    Upload an image and classify it using the four
    transfer-learning CNN models developed for SmartVision AI.
    """
)

st.info(
    "The models recognize 25 selected COCO object classes."
)


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Classification Settings")

model_option = st.sidebar.selectbox(
    "Select model",
    [
        "Compare All Models",
        "ResNet50",
        "EfficientNetB0",
        "VGG16",
        "MobileNetV2"
    ]
)

top_k = st.sidebar.slider(
    "Number of predictions",
    min_value=1,
    max_value=5,
    value=5
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is None:

    st.markdown(
        """
        ### How to use

        1. Upload a JPG, JPEG or PNG image.
        2. Select a CNN model from the sidebar.
        3. Choose how many predictions to display.
        4. SmartVision AI will show the predicted class,
           confidence score and Top-K results.
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


left_col, right_col = st.columns(
    [1, 1.4]
)


with left_col:

    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )


# ============================================================
# PREDICTION HELPER
# ============================================================

def run_model(
    model_name,
    loader_function
):

    with st.spinner(
        f"Running {model_name}..."
    ):

        model, device = loader_function()

        result = classify_image(
            model=model,
            image=image,
            device=device,
            top_k=top_k
        )

    return result


# ============================================================
# SINGLE MODEL MODE
# ============================================================

try:
    if model_option != "Compare All Models":

        loader_map = {
            "VGG16": get_vgg16,
            "ResNet50": get_resnet50,
            "MobileNetV2": get_mobilenetv2,
            "EfficientNetB0": get_efficientnetb0
        }

        result = run_model(
            model_option,
            loader_map[model_option]
        )

        with right_col:

            st.subheader(
                f"{model_option} Prediction"
            )

            st.success(
                f"Prediction: "
                f"{result['predicted_class']}"
            )

            metric1, metric2 = st.columns(2)

            metric1.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            metric2.metric(
                "Inference Time",
                f"{result['latency_ms']:.2f} ms"
            )

            st.subheader(
                f"Top-{top_k} Predictions"
            )

            prediction_df = pd.DataFrame(
                result["top_predictions"]
            )

            prediction_df["confidence"] = (
                prediction_df["confidence"]
                * 100
            )

            prediction_df.columns = [
                "Class",
                "Confidence (%)"
            ]

            st.dataframe(
                prediction_df,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                prediction_df.set_index(
                    "Class"
                )["Confidence (%)"]
            )


    # ============================================================
    # COMPARE ALL MODELS
    # ============================================================

    else:

        st.subheader(
            "Four-Model Comparison"
        )

        loaders = {
            "VGG16": get_vgg16,
            "ResNet50": get_resnet50,
            "MobileNetV2": get_mobilenetv2,
            "EfficientNetB0": get_efficientnetb0
        }

        comparison_results = {}

        for name, loader in loaders.items():

            comparison_results[name] = (
                run_model(
                    name,
                    loader
                )
            )


        # --------------------------------------------------------
        # SUMMARY TABLE
        # --------------------------------------------------------

        summary_rows = []

        for name, result in (
            comparison_results.items()
        ):

            summary_rows.append({
                "Model": name,
                "Prediction":
                    result["predicted_class"],

                "Confidence (%)":
                    result["confidence"] * 100,

                "Inference (ms)":
                    result["latency_ms"]
            })


        summary_df = pd.DataFrame(
            summary_rows
        )

        summary_df[
            "Confidence (%)"
        ] = summary_df[
            "Confidence (%)"
        ].round(2)

        summary_df[
            "Inference (ms)"
        ] = summary_df[
            "Inference (ms)"
        ].round(2)


        with right_col:

            st.subheader(
                "Prediction Summary"
            )

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )


        # --------------------------------------------------------
        # INDIVIDUAL TOP-K RESULTS
        # --------------------------------------------------------

        st.divider()

        st.subheader(
            f"Top-{top_k} Predictions by Model"
        )

        model_columns = st.columns(2)

        for index, (
            name,
            result
        ) in enumerate(
            comparison_results.items()
        ):

            with model_columns[index % 2]:

                st.markdown(
                    f"### {name}"
                )

                st.write(
                    f"**Prediction:** "
                    f"{result['predicted_class']}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{result['confidence'] * 100:.2f}%"
                )

                prediction_df = pd.DataFrame(
                    result["top_predictions"]
                )

                prediction_df[
                    "confidence"
                ] *= 100

                prediction_df.columns = [
                    "Class",
                    "Confidence (%)"
                ]

                st.dataframe(
                    prediction_df,
                    use_container_width=True,
                    hide_index=True
                )
except Exception:
    st.error('Classification failed. Check model files and available memory, then retry.')
    st.stop()


st.divider()

st.caption(
    "SmartVision AI | "
    "Multi-Model Image Classification"
)
