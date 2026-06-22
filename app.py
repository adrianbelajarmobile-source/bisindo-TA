from __future__ import annotations

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.bisindo_app.config import (
    AUTO_SEQUENCE_ON_MOTION,
    DEFAULT_RECOGNITION_MODE,
    MODE_ALPHABET,
    MODE_AUTO,
    MODE_KATA,
    ROUTER_WINDOW_FRAMES,
    SEQUENCE_LENGTH,
    TOKEN_IDLE,
    VALID_RECOGNITION_MODES,
)
from src.bisindo_app.processor import BisindoVideoProcessor
from src.bisindo_app.ui import (
    inject_mobile_css,
    render_header,
    render_mode_title,
    render_outputs,
    render_result_card,
)


def init_session_state() -> None:
    if "camera_facing" not in st.session_state:
        st.session_state.camera_facing = "environment"

    if "camera_version" not in st.session_state:
        st.session_state.camera_version = 0

    if "recognition_mode" not in st.session_state:
        st.session_state.recognition_mode = DEFAULT_RECOGNITION_MODE

    if st.session_state.recognition_mode not in VALID_RECOGNITION_MODES:
        st.session_state.recognition_mode = DEFAULT_RECOGNITION_MODE

def default_ui_state() -> dict:
    return {
        "predicted_letter": "...",
        "raw_prediction": "...",
        "confidence": 0.0,
        "margin": 0.0,
        "hand_count": 0,
        "hasil_kata": "",
        "hasil_kalimat": "",
        "mode": "STATIC",
        "sequence_prediction": "...",
        "sequence_confidence": 0.0,
        "sequence_margin": 0.0,
        "sequence_progress": 0,
        "sequence_trigger_ready": False,
        "auto_sequence_on_motion": AUTO_SEQUENCE_ON_MOTION,
        "motion_score": 0.0,
        "pending_letter": "",
        "pending_letter_frames": 0,
        "hasil_kata_committed_preview": False,
        "last_static_letter_preview": "",
        "recognition_mode": DEFAULT_RECOGNITION_MODE,
        "router_frames": 0,
        "router_window": ROUTER_WINDOW_FRAMES,
        "token_mode": TOKEN_IDLE,
        "sequence_locked_until_release": False,
    }


def resolve_status(state: dict) -> tuple[str, str]:
    hand_count = state["hand_count"]
    mode = state["mode"]
    sequence_progress = state["sequence_progress"]
    recognition_mode = state["recognition_mode"]

    if hand_count == 0:
        return "Siap membaca", "#64748b"

    if mode == "SEQUENCE":
        return f"Membaca kata {sequence_progress}/{SEQUENCE_LENGTH}", "#f5bd45"

    if recognition_mode == MODE_ALPHABET:
        return "Mode alfabet", "#38bdf8"

    if recognition_mode == MODE_KATA:
        return "Mode kata", "#a78bfa"

    return "Mendeteksi otomatis", "#22c55e"


def main() -> None:
    st.set_page_config(
        page_title="Penerjemah BISINDO",
        page_icon="B",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st_autorefresh(interval=1200, key="ui_refresh")

    init_session_state()
    inject_mobile_css()
    render_header()

    mode_label_map = {
        MODE_AUTO: "Auto",
        MODE_ALPHABET: "A-Z",
        MODE_KATA: "Kata",
    }

    mode_options = [MODE_AUTO, MODE_ALPHABET, MODE_KATA]

    render_mode_title()

    control_mode_col, control_camera_col = st.columns([5.8, 0.9], gap="small")

    with control_mode_col:
        selected_mode = st.radio(
            "Mode penerjemah",
            options=mode_options,
            index=mode_options.index(st.session_state.recognition_mode),
            format_func=lambda mode: mode_label_map[mode],
            horizontal=True,
            key="recognition_mode_radio",
            label_visibility="collapsed",
        )

    with control_camera_col:
        switch_camera = st.button(
            "",
            key="switch_camera_button",
            help="Ganti kamera depan/belakang",
            use_container_width=True,
            icon=":material/cameraswitch:",
        )

    if selected_mode != st.session_state.recognition_mode:
        st.session_state.recognition_mode = selected_mode
        st.rerun()

    if switch_camera:
        st.session_state.camera_facing = (
            "environment" if st.session_state.camera_facing == "user" else "user"
        )
        st.session_state.camera_version += 1
        st.rerun()

    ctx = webrtc_streamer(
        key=f"bisindo-realtime-{st.session_state.camera_facing}-{st.session_state.camera_version}",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=BisindoVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 10, "max": 10},
                "facingMode": {"ideal": st.session_state.camera_facing},
            },
            "audio": False,
        },
        async_processing=False,
        desired_playing_state=True,
    )

    if ctx.video_processor:
        ctx.video_processor.set_recognition_mode(st.session_state.recognition_mode)

    state = default_ui_state()

    if ctx.video_processor:
        state = ctx.video_processor.get_ui_state()

    hasil_kata = state["hasil_kata"] if state["hasil_kata"] else "-"
    hasil_kalimat = state["hasil_kalimat"] if state["hasil_kalimat"] else "-"
    status_text, dot_color = resolve_status(state)

    render_result_card(
        state=state,
        status_text=status_text,
    )

    render_outputs(
        hasil_kata=hasil_kata,
        hasil_kalimat=hasil_kalimat,
    )

    col_hapus, col_simpan = st.columns(2, gap="small")

    with col_hapus:
        hapus_clicked = st.button(
            "Hapus",
            key="hapus_button",
            use_container_width=True,
            icon=":material/delete:",
        )

    with col_simpan:
        simpan_clicked = st.button(
            "Simpan",
            key="simpan_button",
            use_container_width=True,
            icon=":material/save:",
        )

    if hapus_clicked:
        if ctx.video_processor:
            ctx.video_processor.clear_text()
        st.rerun()

    if simpan_clicked:
        if ctx.video_processor:
            ctx.video_processor.save_current_word()
        st.rerun()


if __name__ == "__main__":
    main()