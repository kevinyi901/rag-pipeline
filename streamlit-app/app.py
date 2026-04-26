import os, json, time, uuid, re
from datetime import datetime
from typing import Dict, List
import pandas as pd
import requests
import streamlit as st
import logging
from dotenv import load_dotenv

load_dotenv()
try:
    import boto3
    _boto3_available = True
except ImportError:
    _boto3_available = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Eyegentic",
    page_icon="👁",
    layout="wide",
)

VERSION = "3.0.0"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Roboto+Condensed:wght@700&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto', Arial, sans-serif;
}

/* Clean light background for the main content area */
.stApp {
    background-color: #F7F9FB;
    color: #1A1A2E;
}

/* Sidebar — Original Blue, the one place we go deep with brand color */
section[data-testid="stSidebar"] {
    background-color: #003865;
}
section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background-color: transparent;
    color: #FFFFFF !important;
    font-family: 'Roboto', Arial, sans-serif;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.4);
    border-radius: 4px;
    width: 100%;
    margin-top: 0.5rem;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #00C1D5;
    color: #00C1D5 !important;
    background-color: transparent;
}

/* Main title */
h1 {
    font-family: 'Roboto Condensed', Arial, sans-serif !important;
    font-weight: 700 !important;
    color: #003865 !important;
    font-size: 1.8rem !important;
}

h2, h3 {
    font-family: 'Roboto', Arial, sans-serif !important;
    font-weight: 700 !important;
    color: #003865 !important;
}

p, li, label {
    color: #1A1A2E;
    font-size: 16px;
    line-height: 1.6;
}

.stCaption, small {
    color: #6B7280 !important;
    font-size: 13px;
}

/* Chat messages — plain white cards, subtle border */
[data-testid="stChatMessage"] {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    padding: 0.75rem 1rem;
}

/* Tabs — minimal, accent on active */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent;
    border-bottom: 2px solid #E5E7EB;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Roboto', Arial, sans-serif;
    font-weight: 500;
    color: #6B7280;
    background-color: transparent;
    border: none;
    padding: 0.6rem 1.25rem;
}
.stTabs [aria-selected="true"] {
    color: #003865 !important;
    border-bottom: 2px solid #00C1D5 !important;
    background-color: transparent !important;
}

/* Primary button */
.stButton > button[kind="primary"],
.stFormSubmitButton > button {
    background-color: #003865;
    color: #FFFFFF;
    font-family: 'Roboto', Arial, sans-serif;
    font-weight: 700;
    font-size: 15px;
    border: none;
    border-radius: 4px;
    padding: 0.6rem 1.5rem;
}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button:hover {
    background-color: #00C1D5;
    color: #FFFFFF;
}

/* Default buttons */
.stButton > button {
    background-color: #FFFFFF;
    color: #003865;
    font-family: 'Roboto', Arial, sans-serif;
    font-weight: 500;
    border: 1px solid #003865;
    border-radius: 4px;
}
.stButton > button:hover {
    background-color: #003865;
    color: #FFFFFF;
}

/* Text input */
.stTextInput > div > div > input {
    background-color: #FFFFFF;
    color: #1A1A2E;
    border: 1px solid #D1D5DB;
    border-radius: 4px;
    font-family: 'Roboto', Arial, sans-serif;
    font-size: 15px;
}
.stTextInput > div > div > input:focus {
    border-color: #00C1D5;
    box-shadow: 0 0 0 2px rgba(0,193,213,0.15);
}
.stTextInput > div > div > input::placeholder {
    color: #9CA3AF;
}

/* Alert / info boxes */
.stAlert {
    border-left: 3px solid #00C1D5;
    background-color: #EFF9FB;
    color: #1A1A2E;
}

/* Sticky bottom bar */
.ub-bottom {
    position: sticky;
    bottom: 0;
    z-index: 50;
    background: #F7F9FB;
    padding: 0.75rem 0;
    border-top: 1px solid #E5E7EB;
}
.ub-bottom form { margin: 0; }

hr {
    border-color: #E5E7EB;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# EC2 Server Management
# =========================
EC2_INSTANCE_ID = os.getenv("EC2_INSTANCE_ID", "").strip()
AWS_REGION = os.getenv("AWS_REGION", "us-east-1").strip()


def _ec2_client():
    return boto3.client("ec2", region_name=AWS_REGION)


def get_server_state() -> str:
    if not _boto3_available or not EC2_INSTANCE_ID:
        return "unknown"
    try:
        resp = _ec2_client().describe_instances(InstanceIds=[EC2_INSTANCE_ID])
        return resp["Reservations"][0]["Instances"][0]["State"]["Name"]
    except Exception as e:
        logger.warning(f"Could not get EC2 state: {e}")
        return "unknown"


def start_server():
    try:
        _ec2_client().start_instances(InstanceIds=[EC2_INSTANCE_ID])
    except Exception as e:
        st.error(f"Failed to start server: {e}")


def stop_server(hibernate: bool = True):
    try:
        _ec2_client().stop_instances(InstanceIds=[EC2_INSTANCE_ID], Hibernate=hibernate)
    except Exception:
        try:
            _ec2_client().stop_instances(InstanceIds=[EC2_INSTANCE_ID])
        except Exception as e:
            st.error(f"Failed to stop server: {e}")


def check_api_health() -> bool:
    if not API_URL:
        return False
    try:
        health_url = API_URL.rsplit("/", 1)[0] + "/health"
        r = requests.get(health_url, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# =========================
# Backend
# =========================
API_URL = os.getenv("UNBARRED_API", "").strip()
API_KEY = os.getenv("UNBARRED_API_KEY", "").strip()


def call_backend_api(payload: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    r = requests.post(API_URL, json=payload, headers=headers, timeout=180)
    r.raise_for_status()
    return r.json()


# =========================
# Session
# =========================
ss = st.session_state
if "messages" not in ss:
    ss.messages = []
if "run_id" not in ss:
    ss.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:6]

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown(
        "<p style='font-family:Roboto Condensed,Arial,sans-serif; font-weight:700; font-size:1.3rem; color:#FFFFFF; margin-bottom:0;'>Eyegentic</p>",
        unsafe_allow_html=True,
    )
    st.caption(f"Run: {ss.run_id} • v{VERSION}")
    st.markdown("---")
    if st.button("Clear Conversation"):
        ss.messages = []
        st.rerun()



@st.cache_data
def load_filter_options() -> dict:
    try:
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        r = requests.get(f"{API_URL.rstrip('/filters').rstrip('/query')}/filters", headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Could not load filter options: {e}")
        return {}

filter_options = load_filter_options()

# =========================
# Tabs
# =========================
tab_chat, tab_upload = st.tabs(["💬 Chat", "📤 Upload Documents"])

with tab_upload:
    st.write("Upload Documents Interface")

# =========================
# Main
# =========================
st.title("Eyegentic")

# ---- EC2 server gate ----
if EC2_INSTANCE_ID:
    _state = get_server_state()

    if _state == "stopped":
        st.info("The search server is currently offline to save costs.")
        if st.button("Start Server", type="primary", use_container_width=True):
            start_server()
            st.rerun()
        st.caption("Estimated startup time: ~2–3 minutes")
        st.stop()

    elif _state in ("pending", "starting"):
        with st.status("Starting server...", expanded=True):
            st.write("EC2 instance is booting up.")
            st.write("Checking again in 5 seconds...")
        time.sleep(5)
        st.rerun()

    elif _state == "running":
        if not check_api_health():
            with st.status("Server is warming up...", expanded=True):
                st.write("EC2 is running — loading models into GPU memory.")
                st.write("Usually ready within 60–90 seconds.")
            time.sleep(10)
            st.rerun()

    elif _state in ("stopping", "shutting-down"):
        st.warning("The server is shutting down. Refresh in a moment.")
        st.stop()

    elif _state == "unknown":
        if not _boto3_available:
            st.error("boto3 is not installed. Run `pip install boto3` to enable server management.")
            st.stop()
        st.warning("Could not determine server state. Proceeding anyway.")

# Render chat history
for m in ss.messages:
    with st.chat_message(m["role"]):
        content = m["content"]
        if content and m["role"] == "assistant":
            content = content.replace("$", r"\$")
        st.markdown(content)

# ---- Sticky bottom search form ----
st.markdown('<div class="ub-bottom">', unsafe_allow_html=True)
with st.form("ub_search_form", clear_on_submit=True):
    user_text = st.text_input(
        "Ask a question…",
        placeholder="Ask a question about retinal disease research…",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Search", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if submitted and user_text.strip():
    logger.info(f"User Query: '{user_text}' [RunID: {ss.run_id}]")
    ss.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    payload = {"query": user_text}

    with st.spinner("Running search…"):
        try:
            t0 = time.perf_counter()
            data = call_backend_api(payload)
            took_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(f"Search Success in {took_ms}ms [RunID: {ss.run_id}]")
        except requests.HTTPError as e:
            logger.error(f"Backend Error: {str(e)} [RunID: {ss.run_id}]")
            st.error(f"Backend error: {e}\n\n{getattr(e.response, 'text', '')}")
            st.stop()
        except Exception as e:
            logger.error(f"Unexpected Error: {str(e)} [RunID: {ss.run_id}]")
            st.error(str(e))
            st.stop()

    response_text = data.get("response", "")

    with st.chat_message("assistant"):
        if response_text:
            st.markdown(response_text.replace("$", r"\$"))
        st.caption(f"Latency: {took_ms} ms")

    ss.messages.append({"role": "assistant", "content": response_text})