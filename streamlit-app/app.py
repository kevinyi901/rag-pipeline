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

# Setup logging at the top of your file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="EYEgotthis", page_icon="⚖️", layout="wide")
VERSION = "3.0.0"

st.markdown(
    """
<style>
.ub-bottom {
  position: sticky;
  bottom: 0;
  z-index: 50;
  background: var(--background-color);
  padding: 0.5rem 0 0.75rem 0;
  border-top: 1px solid rgba(255,255,255,0.08);
}
.ub-bottom form { margin: 0; }
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
    """Returns the EC2 instance state name, or 'unknown' on error."""
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
    """Returns True if the RAG API /health endpoint responds 200."""
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
    st.header("Eyegotthis")
    st.caption(f"Run: {ss.run_id} • v{VERSION}")
    if st.button("Clear conversation"):
        ss.messages = []
        st.rerun()


# Testing document upload section:

st.set_page_config(
    page_title="FFB Progress Report Assistant",
    page_icon="📄",
    layout="wide",
)


# ── Tabs ───────────────────────────────────────────────────────────────────
tab_chat, tab_upload = st.tabs(["💬 Chat", "📤 Upload Documents"])

with tab_chat:
    st.write("Chat Interface")
with tab_upload:
    st.write("Upload Documents Interface")

# =========================
# Main
# =========================
st.title("EyeSearch")

# ---- EC2 server gate ----
if EC2_INSTANCE_ID:
    _state = get_server_state()

    if _state == "stopped":
        st.info("The search server is currently offline to save costs.")
        if st.button("Start Server", type="primary", use_container_width=True):
            start_server()
            st.rerun()
        st.caption("Estimated startup time: ~2-3 minutes")
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
                st.write("Usually ready within 60-90 seconds.")
            time.sleep(10)
            st.rerun()

    elif _state in ("stopping", "shutting-down"):
        st.warning("The server is shutting down. Refresh in a moment.")
        st.stop()

    elif _state == "unknown":
        if not _boto3_available:
            st.error("boto3 is not installed. Run `pip install boto3` to enable server management.")
            st.stop()
        # EC2_INSTANCE_ID is set but state lookup failed — surface it but don't block
        st.warning("Could not determine server state. Proceeding anyway.")

# Render history
for m in ss.messages:
    with st.chat_message(m["role"]):
        content = m["content"]
        if content and m["role"] == "assistant":
            content = content.replace("$", r"\$")
        st.markdown(content)

# ---- Sticky bottom form ----
st.markdown('<div class="ub-bottom">', unsafe_allow_html=True)
with st.form("ub_search_form", clear_on_submit=True):
    user_text = st.text_input(
        "Ask a question…",
        placeholder="Ask a question…",
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