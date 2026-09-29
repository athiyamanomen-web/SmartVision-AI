
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


# ============================================================
# CLASS NAMES
# ============================================================

# Classification models use ImageFolder alphabetical order
CLASSIFICATION_CLASSES = [
    "airplane",
    "bed",
    "bench",
    "bicycle",
    "bird",
    "bottle",
    "bowl",
    "bus",
    "cake",
    "car",
    "cat",
    "chair",
    "cow",
    "cup",
    "dog",
    "elephant",
    "horse",
    "motorcycle",
    "person",
    "pizza",
    "potted plant",
    "stop sign",
    "traffic light",
    "train",
    "truck"
]


# ============================================================
# CLASSIFICATION PREPROCESSING
# ============================================================

classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# SINGLE CLASSIFIER PREDICTION
# ============================================================

def classify_image(
    model,
    image,
    device,
    top_k=5
):
    """
    Run one classification model on a PIL image.
    Returns top-k predictions and inference latency.
    """

    image = image.convert("RGB")

    tensor = (
        classification_transform(image)
        .unsqueeze(0)
        .to(device)
    )

    start = time.perf_counter()

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    # Synchronize before timing GPU inference
    if device.type == "cuda":
        torch.cuda.synchronize()

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    top_k = min(
        top_k,
        len(CLASSIFICATION_CLASSES)
    )

    scores, indices = torch.topk(
        probabilities,
        k=top_k
    )

    predictions = []

    for score, index in zip(
        scores.cpu().tolist(),
        indices.cpu().tolist()
    ):

        predictions.append({
            "class": CLASSIFICATION_CLASSES[index],
            "confidence": float(score)
        })

    return {
        "predicted_class": predictions[0]["class"],
        "confidence": predictions[0]["confidence"],
        "top_predictions": predictions,
        "latency_ms": latency_ms
    }


# ============================================================
# COMPARE ALL CLASSIFICATION MODELS
# ============================================================

def compare_classifiers(
    models_dict,
    image,
    device,
    top_k=5
):
    """
    Run VGG16, ResNet50, MobileNetV2 and EfficientNetB0
    on the same image.
    """

    results = {}

    for model_name, model in models_dict.items():

        results[model_name] = classify_image(
            model=model,
            image=image,
            device=device,
            top_k=top_k
        )

    return results


# ============================================================
# YOLO OBJECT DETECTION
# ============================================================

def detect_objects(
    yolo_model,
    image,
    confidence_threshold=0.50,
    nms_iou_threshold=0.70,
    device="cpu"
):
    """
    Detect multiple objects with YOLO.

    Ultralytics performs NMS internally using the supplied
    IoU threshold.
    """

    image = image.convert("RGB")

    np_image = np.array(image)

    results = yolo_model.predict(
        source=np_image,
        imgsz=640,
        conf=confidence_threshold,
        iou=nms_iou_threshold,
        device=device,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        width, height = image.size

        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .tolist()
            )

            confidence = float(
                box.conf[0].cpu()
            )

            class_id = int(
                box.cls[0].cpu()
            )

            # Refine / clip coordinates
            x1 = max(
                0,
                min(int(round(x1)), width - 1)
            )

            y1 = max(
                0,
                min(int(round(y1)), height - 1)
            )

            x2 = max(
                0,
                min(int(round(x2)), width)
            )

            y2 = max(
                0,
                min(int(round(y2)), height)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "class_id": class_id,
                "class": yolo_model.names[class_id],
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

    detections.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    return detections


# ============================================================
# OPTIONAL RESNET50 VERIFICATION
# ============================================================

def verify_detections(
    image,
    detections,
    resnet_model,
    device
):
    """
    Verify each YOLO crop using the best classification model.
    YOLO remains the primary detection label.
    """

    verified = []

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        crop = image.crop(
            (x1, y1, x2, y2)
        )

        cnn_result = classify_image(
            model=resnet_model,
            image=crop,
            device=device,
            top_k=5
        )

        item = detection.copy()

        item["cnn_class"] = (
            cnn_result["predicted_class"]
        )

        item["cnn_confidence"] = (
            cnn_result["confidence"]
        )

        item["class_match"] = (
            item["class"]
            == item["cnn_class"]
        )

        verified.append(item)

    return verified


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image,
    detections
):
    """
    Draw bounding boxes, labels and confidence scores.
    """

    output = image.convert("RGB").copy()

    draw = ImageDraw.Draw(output)

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        label = (
            f"{detection['class']} "
            f"{detection['confidence'] * 100:.1f}%"
        )

        if "class_match" in detection:

            if detection["class_match"]:
                label += " | Verified"

            else:
                label += (
                    f" | CNN: "
                    f"{detection['cnn_class']}"
                )

        # Bounding box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        # Text background
        text_bbox = draw.textbbox(
            (x1, y1),
            label
        )

        text_height = (
            text_bbox[3] - text_bbox[1]
        )

        label_y = max(
            0,
            y1 - text_height - 8
        )

        text_bbox = draw.textbbox(
            (x1, label_y),
            label
        )

        draw.rectangle(
            [
                text_bbox[0] - 3,
                text_bbox[1] - 3,
                text_bbox[2] + 3,
                text_bbox[3] + 3
            ],
            fill="red"
        )

        draw.text(
            (x1, label_y),
            label,
            fill="white"
        )

    return output
