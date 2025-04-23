# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 03:32:07 2025

@author: ktrpt
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import mediapipe as mp

st.title("Face Mosaic App using MediaPipe")
st.write("Upload a video, and this app will automatically detect faces and apply mosaic blur.")

# Initialize MediaPipe Face Detection
@st.cache_resource
def load_face_detector():
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

face_detector = load_face_detector()

video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(tempfile.gettempdir(), "output_mosaic.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    progress_bar = st.progress(0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)

                # Apply mosaic
                face_region = frame[y:y+bh, x:x+bw]
                if face_region.size == 0:
                    continue
                mosaic = cv2.resize(face_region, (10, 10))
                mosaic = cv2.resize(mosaic, (bw, bh), interpolation=cv2.INTER_NEAREST)
                frame[y:y+bh, x:x+bw] = mosaic

        out.write(frame)
        frame_idx += 1
        progress_bar.progress(frame_idx / total_frames)

    cap.release()
    out.release()

    st.success("✅ Processing complete!")

    with open(output_path, "rb") as f:
        st.download_button("Download mosaic video", f, file_name="face_mosaic_output.mp4")

    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(output_path):
        os.remove(output_path)
