
import sys
import time
import threading
from pathlib import Path

import av
import cv2
import streamlit as st
import torch

from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer
)


# ============================================================
# PATH SETUP
# ============================================================

APP_DIR = Path(__file__).resolve().parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from utils.model_loader import (
    get_device,
    load_yolo
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Live Webcam Detection | SmartVision AI",
    page_icon="📹",
    layout="wide"
)


# ============================================================
# LOAD YOLO ONCE
# ============================================================

@st.cache_resource
def get_webcam_yolo():
    return load_yolo()


try:
    yolo_model = get_webcam_yolo()
except Exception:
    st.error('Webcam initialization failed. Check camera permissions and model availability.')
    st.stop()

device = get_device(
    prefer_gpu=True
)

yolo_device = (
    0
    if device.type == "cuda"
    else "cpu"
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📹 Live Webcam Detection")

st.write(
    """
    Run real-time object detection directly from your webcam
    using the fine-tuned SmartVision YOLOv8s model.
    """
)

st.info(
    """
    Click **START** below and allow browser camera permission
    when prompted.
    """
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header(
    "Webcam Detection Settings"
)

confidence_threshold = (
    st.sidebar.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="webcam_confidence"
    )
)

nms_iou_threshold = (
    st.sidebar.slider(
        "NMS IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.70,
        step=0.05,
        key="webcam_nms"
    )
)


# ============================================================
# SHARED METRICS
# ============================================================

metrics = {
    "fps": 0.0,
    "latency_ms": 0.0,
    "detections": 0
}

metrics_lock = threading.Lock()


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class SmartVisionVideoProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.previous_time = None


    def recv(self, frame):
        try:
            return self._process_frame(frame)
        except Exception:
            import logging
            logging.exception('Webcam frame processing failed')
            image = frame.to_ndarray(format='bgr24')
            cv2.putText(image, 'Detection failed - restart camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(image, format='bgr24')


    def _process_frame(self, frame):

        # ----------------------------------------------------
        # FRAME -> NUMPY
        # ----------------------------------------------------

        image = frame.to_ndarray(
            format="bgr24"
        )


        # ----------------------------------------------------
        # YOLO INFERENCE
        # ----------------------------------------------------

        start_time = time.perf_counter()

        results = yolo_model.predict(
            source=image,
            imgsz=640,
            conf=confidence_threshold,
            iou=nms_iou_threshold,
            device=yolo_device,
            verbose=False
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


        # ----------------------------------------------------
        # DRAW DETECTIONS
        # ----------------------------------------------------

        result = results[0]

        detection_count = 0

        if result.boxes is not None:

            detection_count = len(
                result.boxes
            )

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

                class_name = (
                    yolo_model.names[
                        class_id
                    ]
                )

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                label = (
                    f"{class_name} "
                    f"{confidence * 100:.1f}%"
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    label,
                    (
                        x1,
                        max(25, y1 - 10)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.perf_counter()

        if self.previous_time is None:

            fps = 0.0

        else:

            frame_time = (
                current_time
                - self.previous_time
            )

            fps = (
                1.0 / frame_time
                if frame_time > 0
                else 0.0
            )

        self.previous_time = current_time


        # ----------------------------------------------------
        # STORE METRICS
        # ----------------------------------------------------

        with metrics_lock:

            metrics["fps"] = fps

            metrics[
                "latency_ms"
            ] = latency_ms

            metrics[
                "detections"
            ] = detection_count


        # ----------------------------------------------------
        # DISPLAY METRICS ON VIDEO
        # ----------------------------------------------------

        cv2.putText(
            image,
            f"FPS: {fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            f"Latency: {latency_ms:.1f} ms",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            image,
            f"Objects: {detection_count}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # RETURN VIDEO FRAME
        # ----------------------------------------------------

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

info1, info2, info3 = st.columns(3)

info1.metric(
    "Model",
    "YOLOv8s"
)

info2.metric(
    "Device",
    str(device).upper()
)

info3.metric(
    "Input Resolution",
    "640"
)


# ============================================================
# WEBCAM STREAM
# ============================================================

st.subheader("Live Detection")

# WebRTC network configuration
ice_servers = [
    {"urls": ["stun:stun.l.google.com:19302"]}
]

try:
    turn = st.secrets.get("turn", {})

    if all(turn.get(k) for k in ("urls", "username", "credential")):
        ice_servers.append({
            "urls": turn["urls"],
            "username": turn["username"],
            "credential": turn["credential"]
        })
except Exception:
    pass



try:
    webrtc_ctx = webrtc_streamer(
        key="smartvision-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": ice_servers
        },


        video_processor_factory=(
            SmartVisionVideoProcessor
        ),

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )
except Exception:
    st.error('Webcam initialization failed. Check camera permissions and model availability.')
    st.stop()


# ============================================================
# INSTRUCTIONS
# ============================================================

st.divider()

st.subheader("How to Use")

st.markdown(
    """
    1. Click **START**.
    2. Allow webcam permission in your browser.
    3. Place supported objects in front of the camera.
    4. YOLOv8s detects objects in real time.
    5. Bounding boxes show class labels and confidence scores.
    6. FPS, latency and object count are displayed directly
       on the live video.
    7. Click **STOP** when finished.
    """
)


# ============================================================
# SUPPORTED CLASSES
# ============================================================

with st.expander(
    "View Supported Classes"
):

    st.write(
        """
        person, bicycle, car, motorcycle, airplane,
        bus, train, truck, traffic light, stop sign,
        bench, bird, cat, dog, horse, cow, elephant,
        bottle, cup, bowl, pizza, cake, chair,
        potted plant, bed
        """
    )


# ============================================================
# PERFORMANCE NOTE
# ============================================================

st.caption(
    "Live FPS depends on GPU/CPU performance, camera "
    "resolution, browser/WebRTC overhead and network conditions."
)
