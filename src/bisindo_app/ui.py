from __future__ import annotations

import html
import subprocess
import streamlit as st
import json
import streamlit.components.v1 as components


def inject_mobile_css() -> None:
    st.markdown(
        """
        <link
            rel="stylesheet"
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css"
        />
        <style>
            #MainMenu, header, footer { visibility: hidden; }

            :root {
                --bg: #100f0b;
                --surface: #18181d;
                --field: #292a31;
                --gold: #f6b638;
                --gold-soft: #51442a;
                --red-soft: #431215;
                --red: #ef4444;
                --muted: #8d9099;
                --text: #f8fafc;
                --border: rgba(255, 255, 255, 0.065);
            }

            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top right, rgba(246,182,56,0.10), transparent 28%),
                    linear-gradient(180deg, #171108 0%, #0b0b0a 74%);
            }

            .block-container {
                max-width: 430px !important;
                padding-top: 0.82rem !important;
                padding-left: 0.95rem !important;
                padding-right: 0.95rem !important;
                padding-bottom: 0.90rem !important;
            }

            html, body, [class*="css"] {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }

            [data-testid="stVerticalBlock"] { gap: 0.90rem !important; }

            .app-header {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                margin-bottom: 0.52rem;
            }

            .app-title {
                color: var(--text);
                font-size: 1.45rem;
                font-weight: 950;
                letter-spacing: -0.04em;
                line-height: 1.05;
            }

            .app-title span {
                color: var(--gold);
                letter-spacing: 0.01em;
            }

            .app-subtitle {
                color: #f1f5f9;
                opacity: 0.88;
                font-size: 0.78rem;
                font-weight: 750;
                margin-top: 0.40rem;
            }

            .info-button-ui {
                width: 42px;
                height: 42px;
                border-radius: 18px;
                background: #1d202a;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #8d95a5;
                font-size: 1.20rem;
                font-weight: 900;
                box-shadow: 0 10px 24px rgba(0,0,0,0.20);
            }

            .mode-title {
                color: var(--gold);
                font-size: 0.64rem;
                letter-spacing: 0.19em;
                font-weight: 950;
                margin: 0.05rem 0 0.12rem 0.15rem;
            }

            div[role="radiogroup"] {
                display: flex;
                gap: 0.30rem;
                background: rgba(17, 14, 12, 0.76);
                border: 1px solid rgba(246,182,56,0.10);
                border-radius: 999px;
                padding: 0.25rem;
                width: 100%;
                min-height: 2.55rem;
                box-shadow: inset 0 0 18px rgba(0,0,0,0.20);
            }

            div[role="radiogroup"] label {
                background: transparent;
                border-radius: 999px;
                color: #e5e7eb !important;
                font-size: 0.74rem !important;
                font-weight: 900 !important;
                min-height: 2.05rem;
                min-width: 5.05rem;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }

            div[role="radiogroup"] label:has(input:checked) {
                background: var(--gold) !important;
                color: #171006 !important;
                box-shadow: 0 8px 18px rgba(246,182,56,0.20);
            }

            div[role="radiogroup"] label p {
                font-size: 0.74rem !important;
                font-weight: 900 !important;
            }

            div[role="radiogroup"] input { display: none; }

            .stButton > button {
                min-height: 2.85rem;
                border-radius: 16px !important;
                font-weight: 950 !important;
                font-size: 0.95rem !important;
                border: none !important;
                box-shadow: 0 10px 18px rgba(0,0,0,0.14) !important;
            }

            div[data-testid="column"] .stButton > button {
                background: rgba(246,182,56,0.10) !important;
                color: var(--gold) !important;
                border: 1px solid rgba(246,182,56,0.18) !important;
            }

            div[data-testid="stHorizontalBlock"]:last-of-type div[data-testid="column"]:first-child .stButton > button {
                background: var(--red-soft) !important;
                color: var(--red) !important;
                border: 1px solid rgba(239,68,68,0.18) !important;
            }

            div[data-testid="stHorizontalBlock"]:last-of-type div[data-testid="column"]:nth-child(2) .stButton > button {
                background: var(--gold) !important;
                color: #171006 !important;
                border: 1px solid rgba(246,182,56,0.18) !important;
            }

            .result-card {
                background: rgba(24,24,29,0.97);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 0.82rem;
                display: grid;
                grid-template-columns: 168px 1fr;
                gap: 0.82rem;
                align-items: center;
                box-shadow: 0 14px 28px rgba(0,0,0,0.22);
                margin-top: 0.1rem;
                margin-bottom: 0.75rem;
            }

            .result-badge {
                min-height: 76px;
                border-radius: 18px;
                background: var(--gold-soft);
                color: var(--gold);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.95rem;
                font-weight: 1000;
                letter-spacing: -0.03em;
                text-transform: uppercase;
            }

            .raw-label {
                color: var(--muted);
                font-size: 0.88rem;
                font-weight: 800;
                margin-bottom: 0.46rem;
            }

            .raw-value {
                color: #d8dce5;
                font-weight: 850;
            }

            .chip-row {
                display: flex;
                gap: 0.45rem;
                flex-wrap: wrap;
            }

            .chip {
                padding: 0.29rem 0.65rem;
                border-radius: 10px;
                background: #252733;
                color: #e5e7eb;
                font-size: 0.76rem;
                font-weight: 900;
            }

            .chip-green {
                background: #11372e;
                color: #69efb5;
            }

            .section-card {
                background: rgba(24,24,29,0.97);
                border: 1px solid var(--border);
                border-radius: 20px;
                padding: 0.95rem;
                box-shadow: 0 14px 28px rgba(0,0,0,0.20);
                margin-bottom: 0.75rem;
            }

            .mini-label {
                color: var(--muted);
                font-size: 0.70rem;
                letter-spacing: 0.18em;
                font-weight: 950;
                margin-bottom: 0.65rem;
            }

            .word-box, .sentence-box {
                background: var(--field);
                color: var(--text);
                border-radius: 14px;
                padding: 0.78rem 0.92rem;
                font-weight: 950;
                word-break: break-word;
            }

            .word-box {
                min-height: 1.50rem;
                font-size: 1.08rem;
                line-height: 1.25;
            }

                        .sentence-box {
                min-height: 3.05rem;
                font-size: 1.02rem;
                line-height: 1.33;
                display: flex;
                align-items: center;
            }

            .sentence-card {
                margin-bottom: 0.75rem;
            }

            .tts-align-spacer {
                height: 1.55rem;
            }

            .tts-button-caption {
                color: var(--muted);
                font-size: 0.62rem;
                font-weight: 900;
                text-align: center;
                margin-top: 0.35rem;
            }

            .stButton > button span[data-testid="stIconMaterial"] {
                font-size: 1.35rem !important;
            }
            

            iframe {
                border-radius: 20px !important;
                background: #151922 !important;
            }

            .stAlert { display: none !important; }

            @media (max-width: 380px) {
                .block-container {
                    padding-left: 0.72rem !important;
                    padding-right: 0.72rem !important;
                }

                .app-title { font-size: 1.28rem; }
                .result-card { grid-template-columns: 1fr; }
                .result-badge { min-height: 62px; font-size: 1.55rem; }
                .sentence-row { grid-template-columns: 1fr 52px; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div>
                <div class="app-title">Penerjemah <span>BISINDO</span></div>
                <div class="app-subtitle">Arahkan tangan ke kamera</div>
            </div>
            <div class="info-button-ui">
                <i class="fa-solid fa-circle-info"></i>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mode_title() -> None:
    st.markdown('<div class="mode-title">MODE PENERJEMAH</div>', unsafe_allow_html=True)


def _safe(value: object) -> str:
    return html.escape(str(value))


def _speak_text(text: str) -> None:
    cleaned = (text or "").strip()
    if not cleaned or cleaned == "-":
        return

    try:
        subprocess.Popen(
            ["say", cleaned],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass


def render_result_card(state: dict, status_text: str) -> None:
    recognition_mode = str(state.get("recognition_mode", "auto")).lower()
    mode = str(state.get("mode", "STATIC")).upper()
    raw_prediction = state.get("raw_prediction", "-") or "-"
    confidence = float(state.get("confidence", 0.0) or 0.0)
    margin = float(state.get("margin", 0.0) or 0.0)

    if mode == "SEQUENCE":
        badge = "KATA"
    elif recognition_mode == "kata":
        badge = "KATA"
    elif recognition_mode == "alphabet":
        badge = "A–Z"
    else:
        badge = "AUTO"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-badge">{_safe(raw_prediction)}</div>
            <div>
                <div class="raw-label">Mode : <span class="raw-value"> {_safe(badge)}</span></div>
                <div class="chip-row">
                    <div class="chip chip-green">Conf {confidence:.2f}</div>
                    <div class="chip">Margin {margin:.2f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_sentence_tts_card(hasil_kalimat: str) -> None:
    cleaned = (hasil_kalimat or "").strip()
    display_text = cleaned if cleaned else "-"
    speak_text = "" if display_text == "-" else display_text

    components.html(
        f"""
        <div class="sentence-card-html">
            <div class="sentence-label-html">HASIL KALIMAT</div>

            <div class="sentence-row-html">
                <div class="sentence-box-html">{_safe(display_text)}</div>

                <button class="tts-button-html" onclick="speakSentence()" aria-label="Bacakan hasil kalimat">
                    <svg width="34" height="34" viewBox="0 0 24 24" fill="none">
                        <path d="M4 9.5V14.5H8L13 19V5L8 9.5H4Z"
                              fill="currentColor"/>
                        <path d="M16 8C17.2 9.1 17.8 10.4 17.8 12C17.8 13.6 17.2 14.9 16 16"
                              stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
                        <path d="M18.5 5.5C20.5 7.3 21.6 9.5 21.6 12C21.6 14.5 20.5 16.7 18.5 18.5"
                              stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
                    </svg>
                </button>
            </div>
        </div>

        <script>
            function speakSentence() {{
                const text = {json.dumps(speak_text)};
                if (!text) return;

                const synth = window.speechSynthesis;
                if (!synth) return;

                synth.cancel();

                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = "id-ID";
                utterance.rate = 0.95;
                utterance.pitch = 1.0;

                synth.speak(utterance);
            }}
        </script>

        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}

            .sentence-card-html {{
                box-sizing: border-box;
                width: 100%;
                background: rgba(24,24,29,0.97);
                border: 1px solid rgba(255, 255, 255, 0.065);
                border-radius: 20px;
                padding: 1.05rem;
                box-shadow: 0 14px 28px rgba(0,0,0,0.20);
            }}

            .sentence-label-html {{
                color: #8d9099;
                font-size: 0.70rem;
                letter-spacing: 0.30em;
                font-weight: 950;
                margin-bottom: 0.80rem;
            }}

            .sentence-row-html {{
                display: grid;
                grid-template-columns: 1fr 68px;
                gap: 1rem;
                align-items: center;
            }}

            .sentence-box-html {{
                min-height: 58px;
                background: #292a31;
                color: #f8fafc;
                border-radius: 14px;
                padding: 0 1rem;
                font-size: 1.05rem;
                line-height: 1.3;
                font-weight: 950;
                display: flex;
                align-items: center;
                word-break: break-word;
            }}

            .tts-button-html {{
                width: 68px;
                height: 58px;
                border: none;
                border-radius: 18px;
                background: #5a4a24;
                color: #f6b638;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 10px 18px rgba(0,0,0,0.16);
            }}

            .tts-button-html:active {{
                transform: scale(0.97);
            }}

            @media (max-width: 380px) {{
                .sentence-card-html {{
                    padding: 0.95rem;
                }}

                .sentence-row-html {{
                    grid-template-columns: 1fr 58px;
                    gap: 0.75rem;
                }}

                .tts-button-html {{
                    width: 58px;
                    height: 56px;
                    border-radius: 16px;
                }}

                .sentence-box-html {{
                    min-height: 56px;
                    font-size: 0.98rem;
                }}
            }}
        </style>
        """,
        height=136,
        scrolling=False,
    )

def render_outputs(hasil_kata: str, hasil_kalimat: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="mini-label">HASIL KATA</div>
            <div class="word-box">{_safe(hasil_kata)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sentence_tts_card(hasil_kalimat)

# Backward compatible untuk app lama.
def render_status_and_outputs(status_text: str, dot_color: str, hasil_kata: str, hasil_kalimat: str) -> None:
    render_outputs(hasil_kata=hasil_kata, hasil_kalimat=hasil_kalimat)
