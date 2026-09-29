
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

CLASSIFICATION_DIR = (
    PROJECT_DIR
    / "results"
    / "classification"
)

DETECTION_DIR = (
    PROJECT_DIR
    / "results"
    / "detection"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Model Performance | SmartVision AI",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("📊 Model Performance")

st.write(
    """
    Explore the final evaluation results for the SmartVision
    image-classification and object-detection models.
    """
)


# ============================================================
# LOAD CLASSIFICATION RESULTS
# ============================================================

comparison_path = (
    CLASSIFICATION_DIR
    / "final_model_comparison.csv"
)

classification_df = pd.read_csv(
    comparison_path
)


# ============================================================
# CLASSIFICATION SECTION
# ============================================================

st.header("🖼️ Classification Performance")

st.write(
    """
    Four transfer-learning CNN architectures were evaluated
    on the same held-out classification test set.
    """
)


# ------------------------------------------------------------
# BEST MODEL SUMMARY
# ------------------------------------------------------------

best_row = classification_df.loc[
    classification_df[
        "Accuracy (%)"
    ].idxmax()
]

metric1, metric2, metric3, metric4 = (
    st.columns(4)
)

metric1.metric(
    "Selected Classifier",
    best_row["Model"]
)

metric2.metric(
    "Test Accuracy",
    f"{best_row['Accuracy (%)']:.2f}%"
)

metric3.metric(
    "F1 Score",
    f"{best_row['F1 Score (%)']:.2f}%"
)

metric4.metric(
    "Top-5 Accuracy",
    f"{best_row['Top-5 Accuracy (%)']:.2f}%"
)


# ------------------------------------------------------------
# CLASSIFICATION TABLE
# ------------------------------------------------------------

st.subheader("Model Comparison")

display_df = classification_df.copy()

numeric_columns = display_df.select_dtypes(
    include="number"
).columns

display_df[numeric_columns] = (
    display_df[numeric_columns]
    .round(2)
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# ACCURACY CHART
# ------------------------------------------------------------

st.subheader("Test Accuracy Comparison")

accuracy_chart = (
    classification_df[
        [
            "Model",
            "Accuracy (%)"
        ]
    ]
    .set_index("Model")
)

st.bar_chart(
    accuracy_chart
)


# ------------------------------------------------------------
# TRADE-OFF VISUALIZATIONS
# ------------------------------------------------------------

st.subheader(
    "Accuracy, Speed and Model Size Trade-offs"
)

trade_col1, trade_col2 = st.columns(2)

accuracy_speed_path = (
    CLASSIFICATION_DIR
    / "final_accuracy_speed_tradeoff.png"
)

accuracy_size_path = (
    CLASSIFICATION_DIR
    / "final_accuracy_size_tradeoff.png"
)


with trade_col1:

    st.markdown(
        "#### Accuracy vs Inference Speed"
    )

    if accuracy_speed_path.exists():

        st.image(
            str(accuracy_speed_path),
            use_container_width=True
        )


with trade_col2:

    st.markdown(
        "#### Accuracy vs Model Size"
    )

    if accuracy_size_path.exists():

        st.image(
            str(accuracy_size_path),
            use_container_width=True
        )


# ============================================================
# CONFUSION MATRICES
# ============================================================

st.subheader("Confusion Matrices")

confusion_paths = {
    "VGG16":
        CLASSIFICATION_DIR
        / "vgg16_final_confusion_matrix.png",

    "ResNet50":
        CLASSIFICATION_DIR
        / "resnet50_final_confusion_matrix.png",

    "MobileNetV2":
        CLASSIFICATION_DIR
        / "mobilenetv2_final_confusion_matrix.png",

    "EfficientNetB0":
        CLASSIFICATION_DIR
        / "efficientnetb0_final_confusion_matrix.png"
}


selected_confusion_model = (
    st.selectbox(
        "Select classification model",
        list(confusion_paths.keys()),
        index=1
    )
)

selected_confusion_path = (
    confusion_paths[
        selected_confusion_model
    ]
)

if selected_confusion_path.exists():

    st.image(
        str(selected_confusion_path),
        caption=(
            f"{selected_confusion_model} "
            "Confusion Matrix"
        ),
        use_container_width=True
    )


# ============================================================
# TRAINING CURVES
# ============================================================

with st.expander(
    "View Training Curves"
):

    selected_curve_model = (
        st.selectbox(
            "Select model for training curves",
            [
                "VGG16",
                "ResNet50",
                "MobileNetV2",
                "EfficientNetB0"
            ]
        )
    )

    curve_prefix = {
        "VGG16": "vgg16",
        "ResNet50": "resnet50",
        "MobileNetV2": "mobilenetv2",
        "EfficientNetB0": "efficientnetb0"
    }[
        selected_curve_model
    ]

    accuracy_curve = (
        CLASSIFICATION_DIR
        / f"{curve_prefix}_final_accuracy_curve.png"
    )

    loss_curve = (
        CLASSIFICATION_DIR
        / f"{curve_prefix}_final_loss_curve.png"
    )

    curve_col1, curve_col2 = (
        st.columns(2)
    )

    with curve_col1:

        if accuracy_curve.exists():

            st.image(
                str(accuracy_curve),
                caption="Accuracy Curve",
                use_container_width=True
            )

    with curve_col2:

        if loss_curve.exists():

            st.image(
                str(loss_curve),
                caption="Loss Curve",
                use_container_width=True
            )


# ============================================================
# OBJECT DETECTION SECTION
# ============================================================

st.divider()

st.header("🎯 Object Detection Performance")

YOLO_METRICS_PATH = (
    DETECTION_DIR
    / "yolov8s_v2_final_test_metrics.json"
)

with open(
    YOLO_METRICS_PATH,
    "r",
    encoding="utf-8"
) as file:

    yolo_metrics = json.load(file)


det1, det2, det3, det4 = (
    st.columns(4)
)

det1.metric(
    "Precision",
    f"{yolo_metrics['precision'] * 100:.2f}%"
)

det2.metric(
    "Recall",
    f"{yolo_metrics['recall'] * 100:.2f}%"
)

det3.metric(
    "mAP@0.5",
    f"{yolo_metrics['map50'] * 100:.2f}%"
)

det4.metric(
    "mAP@0.5:0.95",
    f"{yolo_metrics['map50_95'] * 100:.2f}%"
)


st.write(
    f"""
    **Test images:** {yolo_metrics['test_images']}  
    **Test object instances:** {yolo_metrics['test_instances']}  
    **Previous V1 mAP@0.5:** {yolo_metrics['previous_v1_map50'] * 100:.2f}%  
    **V2 improvement:** {yolo_metrics['map50_change_percentage_points']:.2f}
    percentage points
    """
)



# ============================================================
# CLASS-WISE DETECTION PERFORMANCE
# ============================================================

st.subheader("Class-wise Detection Performance")

st.write(
    """
    The table and chart below show the final YOLOv8s V2
    **mAP@0.5 for each of the 25 supported object classes**.
    This helps identify classes where the detector performs
    strongly and classes that remain more challenging.
    """
)


classwise_map50 = {
    "person": 0.815,
    "bicycle": 0.814,
    "car": 0.741,
    "motorcycle": 0.846,
    "airplane": 0.889,
    "bus": 0.954,
    "train": 0.929,
    "truck": 0.726,
    "traffic light": 0.629,
    "stop sign": 0.913,
    "bench": 0.316,
    "bird": 0.547,
    "cat": 0.995,
    "dog": 0.962,
    "horse": 0.981,
    "cow": 0.934,
    "elephant": 0.966,
    "bottle": 0.599,
    "cup": 0.809,
    "bowl": 0.786,
    "pizza": 0.883,
    "cake": 0.875,
    "chair": 0.663,
    "potted plant": 0.698,
    "bed": 0.924
}


classwise_df = pd.DataFrame(
    {
        "Class": list(
            classwise_map50.keys()
        ),
        "mAP@0.5 (%)": [
            value * 100
            for value in classwise_map50.values()
        ]
    }
)


classwise_df = (
    classwise_df
    .sort_values(
        "mAP@0.5 (%)",
        ascending=False
    )
    .reset_index(drop=True)
)


classwise_df.insert(
    0,
    "Rank",
    range(
        1,
        len(classwise_df) + 1
    )
)


class_col1, class_col2 = st.columns(
    [1, 1.4]
)


with class_col1:

    st.markdown(
        "#### Per-Class mAP@0.5"
    )

    st.dataframe(
        classwise_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "mAP@0.5 (%)":
                st.column_config.NumberColumn(
                    "mAP@0.5 (%)",
                    format="%.2f%%"
                )
        }
    )


with class_col2:

    st.markdown(
        "#### mAP@0.5 by Class"
    )

    import matplotlib.pyplot as plt

    chart_df = (
        classwise_df[
            ["Class", "mAP@0.5 (%)"]
        ]
        .sort_values(
            "mAP@0.5 (%)",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 10)
    )

    ax.barh(
        chart_df["Class"],
        chart_df["mAP@0.5 (%)"]
    )

    ax.set_xlabel("mAP@0.5 (%)")
    ax.set_ylabel("Object Class")
    ax.set_xlim(0, 100)

    ax.set_title(
        "YOLOv8s V2 Class-wise Detection Performance"
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


best_class = classwise_df.iloc[0]
weakest_class = classwise_df.iloc[-1]


summary1, summary2 = st.columns(2)


summary1.metric(
    "Highest Class mAP@0.5",
    (
        f"{best_class['Class']} — "
        f"{best_class['mAP@0.5 (%)']:.2f}%"
    )
)


summary2.metric(
    "Lowest Class mAP@0.5",
    (
        f"{weakest_class['Class']} — "
        f"{weakest_class['mAP@0.5 (%)']:.2f}%"
    )
)


st.caption(
    "Per-class values are from the final YOLOv8s V2 "
    "test evaluation."
)



# ============================================================
# DETECTION VISUALIZATIONS
# ============================================================

st.subheader("Detection Examples")

prediction_path = (
    DETECTION_DIR
    / "yolov8s_v2_final_predictions.png"
)

failure_path = (
    DETECTION_DIR
    / "yolov8s_v2_final_failure_cases.png"
)


visual_col1, visual_col2 = st.columns(2)


with visual_col1:

    st.markdown(
        "#### Prediction Examples"
    )

    if prediction_path.exists():

        st.image(
            str(prediction_path),
            use_container_width=True
        )


with visual_col2:

    st.markdown(
        "#### Failure Case Analysis"
    )

    if failure_path.exists():

        st.image(
            str(failure_path),
            use_container_width=True
        )


# ============================================================
# MODEL SELECTION SUMMARY
# ============================================================

st.divider()

st.header("🏆 Final Model Selection")

st.markdown(
    """
    **Classification:** ResNet50 was selected as the primary
    classification model based on the final held-out test
    evaluation.

    **Object Detection:** The final targeted YOLOv8s V2
    checkpoint is used for multi-object detection.

    Quantization was also evaluated during integration.
    Dynamic INT8 preserved the tested prediction but did not
    improve CPU latency on the development machine, so the
    FP32 ResNet50 model was retained for deployment.
    """
)


st.divider()

st.caption(
    "SmartVision AI | Model Evaluation Dashboard"
)
