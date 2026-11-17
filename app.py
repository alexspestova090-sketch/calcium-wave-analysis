import tempfile
import cv2
import numpy as np
import streamlit as st
from event_hierarchy import classify_event, TrueWave, Artifact

st.set_page_config(page_title="Анализ кальциевых волн", layout="wide")
st.title("Анализатор кальциевых волн")

st.sidebar.header("Параметры фильтрации")
brightness_thresh = st.sidebar.slider("Порог яркости", min_value=1, max_value=100, value=10, step=1)
min_area = st.sidebar.slider("Мин. площадь волны (px)", min_value=100, max_value=10000, value=2000, step=100)
min_duration = st.sidebar.slider("Мин. длительность (сек)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
min_straightness = st.sidebar.slider("Мин. прямолинейность (0..1)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
blur_sigma = st.sidebar.slider("Радиус сглаживания фона (sigma)", min_value=10.0, max_value=100.0, value=50.0, step=5.0)

uploaded_file = st.file_uploader("Загрузите видеофайл (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_idx = st.slider("Выберите кадр для просмотра результата", 0, max(0, total_frames - 1), min(50, max(0, total_frames - 1)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
        local = cv2.subtract(gray, blur)

        _, mask = cv2.threshold(local, brightness_thresh, 255, cv2.THRESH_BINARY)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vis_frame = frame.copy()
        events = []

        for idx, c in enumerate(contours, start=1):
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            cx = (M["m10"] / M["m00"]) if M["m00"] > 0 else 0.0
            cy = (M["m01"] / M["m00"]) if M["m00"] > 0 else 0.0

            est_duration = (area / 2500.0) * 0.8
            demo_traj = [(cx, cy), (cx + 15.0, cy + 15.0)] if area >= min_area else [(cx, cy), (cx + 1.0, cy - 1.0), (cx, cy)]

            ev = classify_event(
                event_id=idx,
                start_frame=frame_idx,
                end_frame=frame_idx + int(est_duration * fps),
                duration_sec=est_duration,
                peak_area_px=area,
                trajectory=demo_traj,
                min_area_px=min_area,
                min_duration_sec=min_duration,
                min_straightness=min_straightness,
            )
            events.append(ev)

            if isinstance(ev, TrueWave):
                cv2.drawContours(vis_frame, [c], -1, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Wave #{idx}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif isinstance(ev, Artifact):
                cv2.drawContours(vis_frame, [c], -1, (0, 0, 255), 1)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Исходный кадр")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader("Результат детекции и фильтрации")
            st.image(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("Результаты классификации")
        true_waves = [e for e in events if e.is_valid_wave()]
        artifacts = [e for e in events if not e.is_valid_wave()]

        m1, m2, m3 = st.columns(3)
        m1.metric("Всего контуров", len(events))
        m2.metric("Настоящие волны", len(true_waves))
        m3.metric("Отсеяно артефактов", len(artifacts))

        for ev in events:
            if ev.is_valid_wave():
                st.success(ev.summary())
            else:
                st.warning(ev.summary())
