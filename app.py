import os
import tempfile
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from ultralytics import YOLO

st.set_page_config(page_title="AnatoScope AI", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }
    html, body, .stApp {
        background: radial-gradient(circle at top left, rgba(79, 172, 254, 0.18), transparent 24%),
                    linear-gradient(135deg, #050816 0%, #0b1430 45%, #111c3a 100%);
        color: #f8fbff;
        font-family: "Inter", "Segoe UI", sans-serif;
    }
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    [data-testid="stSidebar"] {
        background: rgba(5, 10, 24, 0.95);
        backdrop-filter: blur(18px);
        border-right: 1px solid rgba(79, 172, 254, 0.28);
        box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.05);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stThumb"] {
        background: linear-gradient(135deg, #00f2fe, #4facfe);
        border: 2px solid #ffffff;
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stFilledTrack"] {
        background: linear-gradient(90deg, #00f2fe, #4facfe);
    }
    .hero-card {
        padding: 1.4rem 1.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(8, 16, 34, 0.94), rgba(13, 26, 62, 0.86));
        border: 1px solid rgba(79, 172, 254, 0.35);
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.04) inset, 0 16px 50px rgba(0, 242, 254, 0.12);
        margin-bottom: 1rem;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        background: rgba(0, 242, 254, 0.12);
        color: #7aefff;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: 1px solid rgba(0, 242, 254, 0.25);
        box-shadow: 0 0 18px rgba(0, 242, 254, 0.2);
        animation: pulse 2.8s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: translateY(0px); box-shadow: 0 0 18px rgba(0, 242, 254, 0.16); }
        50% { transform: translateY(-1px); box-shadow: 0 0 24px rgba(79, 172, 254, 0.28); }
    }
    .hero-card h1 {
        margin: 0.6rem 0 0.4rem;
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .hero-card p {
        color: #d7e7ff;
        margin: 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .glass-card {
        padding: 1rem 1.1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(10, 20, 40, 0.92), rgba(20, 32, 62, 0.86));
        border: 1px solid rgba(79, 172, 254, 0.28);
        box-shadow: 0 12px 40px rgba(0, 8, 24, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(7, 16, 34, 0.96), rgba(15, 29, 58, 0.9));
        border: 1px solid rgba(79, 172, 254, 0.26);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.24);
        min-height: 118px;
    }
    .metric-card .metric-icon {
        font-size: 1.25rem;
        margin-bottom: 0.45rem;
    }
    .metric-card .metric-title {
        font-size: 0.8rem;
        color: #8bb7ff;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }
    .metric-card .metric-value {
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-card .metric-subtitle {
        font-size: 0.9rem;
        color: #c5d8ff;
        margin-top: 0.2rem;
    }
    .upload-shell {
        padding: 1rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(8, 16, 34, 0.9), rgba(18, 30, 62, 0.84));
        border: 1px dashed rgba(0, 242, 254, 0.45);
        box-shadow: 0 0 28px rgba(79, 172, 254, 0.16);
        margin-bottom: 1rem;
    }
    [data-testid="stFileUploader"] > section {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(79, 172, 254, 0.25);
        border-radius: 16px;
        padding: 0.8rem;
    }
    .stButton > button {
        border: none;
        border-radius: 999px;
        padding: 0.62rem 1.15rem;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #05101d;
        font-weight: 700;
        box-shadow: 0 10px 24px rgba(79, 172, 254, 0.24);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(0, 242, 254, 0.3);
    }
    .stDownloadButton > button {
        border-radius: 999px;
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: #04111d;
        font-weight: 700;
    }
    .stTabs [data-testid="stBaseButton-header"] {
        color: #cfe6ff !important;
    }
    .stTabs [data-testid="stBaseButton-header"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #00f2fe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_metric_card(title, value, icon, subtitle):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    anatomy_path = "anatomy_best.pt"
    tools_path = "tools_best.pt"

    if not os.path.exists(anatomy_path) or not os.path.exists(tools_path):
        raise FileNotFoundError("Expected model files anatomy_best.pt and tools_best.pt were not found.")

    anatomy_model = YOLO(anatomy_path).to(device)
    tools_model = YOLO(tools_path).to(device)
    return anatomy_model, tools_model, device


try:
    anatomy_model, tools_model, device = load_models()
    use_half = device == "cuda"
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()


def resize_mask_to_frame(mask, frame_shape):
    if mask is None:
        return None

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            return None

        mask_arr = mask.astype(np.uint8)
        if mask_arr.shape != frame_shape:
            mask_arr = cv2.resize(mask_arr, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask_arr > 0

    if isinstance(mask, (list, tuple)) and len(mask) > 0 and isinstance(mask[0], (list, tuple, np.ndarray)):
        canvas = np.zeros(frame_shape, dtype=np.uint8)
        points = np.array(mask, dtype=np.int32).reshape(-1, 1, 2)
        if len(points) >= 3:
            cv2.fillPoly(canvas, [points], 1)
        return canvas.astype(bool)

    return None


def iter_mask_candidates(masks):
    if masks is None:
        return []
    if hasattr(masks, "data"):
        return list(masks.data)
    if hasattr(masks, "xy"):
        return list(masks.xy)
    return []


def check_mask_collision(mask_a, mask_b, target_shape):
    if mask_a is None or mask_b is None:
        return False

    resized_a = resize_mask_to_frame(mask_a, target_shape)
    resized_b = resize_mask_to_frame(mask_b, target_shape)
    if resized_a is None or resized_b is None:
        return False

    return bool(np.any(np.logical_and(resized_a, resized_b)))


def build_pdf_report(summary_dict, warnings_list, total_frames, fps, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=18, alignment=1, textColor=colors.HexColor("#E2E8F0"))
    story.append(Paragraph("<b>AnatoScope AI Surgical Report</b>", title_style))
    story.append(Spacer(1, 10))

    duration_sec = round(total_frames / (fps if fps > 0 else 25), 2)
    story.append(Paragraph(f"<b>Video duration:</b> {duration_sec}s ({total_frames} frames)", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Detected anatomy and tools</b>", styles["Heading2"]))
    table_data = [["Element", "Frames detected", "Presence ratio"]]
    for item, count in sorted(summary_dict.items()):
        ratio = f"{round((count / total_frames) * 100, 1)}%" if total_frames > 0 else "0%"
        table_data.append([item, str(count), ratio])

    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.7, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>Safety warnings</b>", styles["Heading2"]))
    if warnings_list:
        for warning in warnings_list:
            story.append(Paragraph(f"• {warning}", styles["BodyText"]))
    else:
        story.append(Paragraph("No critical collisions detected.", styles["BodyText"]))

    doc.build(story)
    return output_path


def get_medical_chat_response(prompt):
    lower_prompt = prompt.lower()
    if any(term in lower_prompt for term in ["bleeding", "hemorrhage", "blood loss"]):
        return "Bleeding control is a priority. Confirm hemodynamic stability, check for visible vessel injury, review the operative field, and escalate to the surgeon or anesthesia team if there is brisk blood loss or instability."
    if any(term in lower_prompt for term in ["infection", "sterile", "contamination"]):
        return "Infection prevention should focus on aseptic technique, sterile field integrity, hand hygiene, and timely antibiotic prophylaxis per protocol. Reassess the field if contamination is suspected."
    if any(term in lower_prompt for term in ["anesthesia", "sedation", "airway"]):
        return "Airway and anesthesia safety require continuous monitoring of oxygenation, ventilation, hemodynamics, and patient response. Any sudden deterioration should trigger immediate escalation and airway reassessment."
    if any(term in lower_prompt for term in ["recovery", "postop", "pain"]):
        return "Postoperative recovery should include pain control, monitoring, early mobilization when appropriate, and prompt review of any new neurologic, respiratory, or cardiovascular symptoms."
    if any(term in lower_prompt for term in ["suture", "staple", "incision"]):
        return "Wound closure planning should balance tissue tension, hemostasis, and the expected healing environment. Follow institutional closure guidance and inspect the incision for signs of dehiscence or ischemia."

    return "This is educational guidance only and does not replace clinical judgment. Please verify any recommendation with the supervising surgeon, anesthesia team, or institutional protocol before acting."


def analyze_video(uploaded_file, conf_threshold, frame_skip):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise RuntimeError("The uploaded video could not be opened.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_width = 640
    target_height = int(height * (target_width / width)) if width > 0 else 360

    output_video = os.path.join(tempfile.gettempdir(), "surgical_analysis_output.mp4")
    # استخدام الترميز المتوافق كلياً مع المتصفحات (H264 / avc1 / mp4v)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_video, fourcc, fps, (target_width, target_height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, (target_width, target_height))

    summary_counter = Counter()
    warnings_log = []
    unique_warnings = set()
    processed_frames = 0

    with torch.inference_mode():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1
            if processed_frames % frame_skip != 0:
                continue

            resized_frame = cv2.resize(frame, (target_width, target_height))
            frame_shape = resized_frame.shape[:2]
            tools_res = tools_model.track(
                resized_frame,
                conf=conf_threshold,
                imgsz=320,
                half=use_half,
                persist=True,
                stream=False,
                verbose=False,
            )[0]
            anatomy_res = anatomy_model(
                resized_frame,
                conf=conf_threshold,
                imgsz=320,
                half=use_half,
                stream=False,
                verbose=False,
            )[0]

            annotated_frame = resized_frame.copy()
            for mask in iter_mask_candidates(anatomy_res.masks):
                mask_arr = resize_mask_to_frame(mask, frame_shape)
                if mask_arr is not None:
                    annotated_frame[mask_arr] = [0, 255, 0]

            if tools_res.boxes is not None:
                annotated_frame = tools_res.plot(img=annotated_frame, conf=False)

            if anatomy_res.boxes is not None:
                anatomy_classes = [anatomy_model.names[int(cls)] for cls in anatomy_res.boxes.cls]
                summary_counter.update(anatomy_classes)
            if tools_res.boxes is not None:
                tools_classes = [tools_model.names[int(cls)] for cls in tools_res.boxes.cls]
                summary_counter.update(tools_classes)

            if tools_res.masks is not None and anatomy_res.masks is not None:
                tool_masks = iter_mask_candidates(tools_res.masks)
                organ_masks = iter_mask_candidates(anatomy_res.masks)
                for tool_index, tool_mask in enumerate(tool_masks):
                    tool_name = tools_model.names[int(tools_res.boxes.cls[tool_index])] if tools_res.boxes is not None else "Tool"
                    tool_id = int(tools_res.boxes.id[tool_index]) if (tools_res.boxes is not None and tools_res.boxes.id is not None) else tool_index
                    for organ_index, organ_mask in enumerate(organ_masks):
                        organ_name = anatomy_model.names[int(anatomy_res.boxes.cls[organ_index])] if anatomy_res.boxes is not None else "Organ"
                        if check_mask_collision(tool_mask, organ_mask, frame_shape):
                            time_sec = round(processed_frames / fps, 1)
                            warning = f"Collision at {time_sec}s: Tool #{tool_id} ({tool_name}) touched {organ_name}"
                            if warning not in unique_warnings:
                                unique_warnings.add(warning)
                                warnings_log.append(warning)

            out.write(annotated_frame)

    cap.release()
    out.release()
    os.remove(temp_path)

    report_path = os.path.join(tempfile.gettempdir(), "anatoscope_report.pdf")
    build_pdf_report(dict(summary_counter), warnings_log, total_frames, fps, report_path)
    return output_video, dict(summary_counter), warnings_log, report_path


# --- Session State Initializations ---
if "medical_chat_messages" not in st.session_state:
    st.session_state.medical_chat_messages = [
        {
            "role": "assistant",
            "content": "Hello, I’m AnatoScope’s surgical AI assistant. Ask me anything about surgical procedures, risks, or recovery.",
        }
    ]

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "analysis_output_video" not in st.session_state:
    st.session_state.analysis_output_video = None
if "analysis_summary" not in st.session_state:
    st.session_state.analysis_summary = {}
if "analysis_warnings" not in st.session_state:
    st.session_state.analysis_warnings = []
if "analysis_report_path" not in st.session_state:
    st.session_state.analysis_report_path = None
if "analysis_metrics" not in st.session_state:
    st.session_state.analysis_metrics = []
if "analysis_uploaded_name" not in st.session_state:
    st.session_state.analysis_uploaded_name = None

# --- Sidebar ---
with st.sidebar:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.header("⚙️ Analysis controls")
    confidence = st.slider("Confidence threshold", 0.1, 0.95, 0.30, 0.05)
    frame_skip = st.slider("Frame skip", 1, 8, 2)
    st.caption("Lower values produce more detail, while higher values speed up analysis.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🩺 Medical Chatbot")
    st.caption("Ask for quick surgical guidance or perioperative safety reminders.")
    for message in st.session_state.medical_chat_messages:
        with st.sidebar.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.sidebar.chat_input("Ask about surgery, anesthesia, or recovery")
    if prompt:
        st.session_state.medical_chat_messages.append({"role": "user", "content": prompt})
        with st.sidebar.chat_message("user"):
            st.write(prompt)

        response = get_medical_chat_response(prompt)
        st.session_state.medical_chat_messages.append({"role": "assistant", "content": response})
        with st.sidebar.chat_message("assistant"):
            st.write(response)
    st.markdown("</div>", unsafe_allow_html=True)


# --- Main Dashboard ---
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">● AI Powered Surgical Inspection</div>
        <h1>AnatoScope AI</h1>
        <p>Realtime surgical scene perception, anatomy recognition, tool tracking, and collision-aware reporting for high-stakes clinical review.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs Navigation
main_tab1, main_tab2 = st.tabs(["📁 Video File Analysis", "📹 Live Endoscope / Camera Feed"])

with main_tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Tools Tracked", "Ready", "🛠️", "Uploads begin a live scan")
    with col2:
        render_metric_card("Safety Score", "98%", "🩺", "Collision monitoring active")
    with col3:
        render_metric_card("Frame Rate", "24 fps", "⚡", "Optimized stream processing")
    with col4:
        render_metric_card("Detections", "0", "👁️", "Awaiting first analysis")

    st.markdown("<div class='upload-shell'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a surgery video", type=["mp4", "avi", "mov"], key="video_uploader")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        if st.session_state.analysis_uploaded_name != uploaded_file.name:
            st.session_state.analysis_done = False
            st.session_state.analysis_output_video = None
            st.session_state.analysis_summary = {}
            st.session_state.analysis_warnings = []
            st.session_state.analysis_report_path = None
            st.session_state.analysis_metrics = []
            st.session_state.analysis_uploaded_name = uploaded_file.name

        st.markdown(
            f"""
            <div class="glass-card">
                <strong>Loaded:</strong> {uploaded_file.name} ({round(uploaded_file.size / (1024 * 1024), 2)} MB)
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Run surgical analysis", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            try:
                with st.spinner("Running deep surgical inspection..."):
                    output_video, summary, warnings, report_path = analyze_video(uploaded_file, confidence, frame_skip)
                progress.progress(1.0)
                status.success("Analysis complete")

                total_detections = sum(summary.values())
                safety_score = max(100 - len(warnings) * 12, 0)
                st.session_state.analysis_done = True
                st.session_state.analysis_output_video = output_video
                st.session_state.analysis_summary = summary
                st.session_state.analysis_warnings = warnings
                st.session_state.analysis_report_path = report_path
                st.session_state.analysis_metrics = [
                    ("Tools Tracked", str(len(summary)), "🛠️", "Unique anatomy and tool classes detected"),
                    ("Safety Score", f"{safety_score}%", "🩺", "Collision risk after inspection"),
                    (
                        "Frame Rate",
                        f"{device.upper()} · {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}",
                        "⚡",
                        "Processing backend",
                    ),
                    ("Detections", str(total_detections), "👁️", "Instance signals across sampled frames"),
                ]
            except Exception as exc:
                st.session_state.analysis_done = False
                st.error(f"Analysis failed: {exc}")

    if st.session_state.analysis_done and st.session_state.analysis_report_path is not None:
        metric_cols = st.columns(4)
        for col, metric in zip(metric_cols, st.session_state.analysis_metrics):
            with col:
                render_metric_card(*metric)

        video_tab, analytics_tab, report_tab = st.tabs(["Video Inspection", "Real-Time Analytics", "Export & Safety Reports"])

        with video_tab:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Annotated video")
            
            # قراءة الملف كبيانات ثنائية لتجنب مشكلة الشاشة السوداء في المتصفح
            if os.path.exists(st.session_state.analysis_output_video):
                with open(st.session_state.analysis_output_video, "rb") as vid_file:
                    video_bytes = vid_file.read()
                st.video(video_bytes, format="video/mp4")
            else:
                st.error("Output video file not found.")
                
            st.markdown("</div>", unsafe_allow_html=True)

        with analytics_tab:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Detection breakdown")
            summary_df = pd.DataFrame([{"Element": key, "Frames detected": value} for key, value in sorted(st.session_state.analysis_summary.items())])
            st.dataframe(summary_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with report_tab:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Safety warnings")
            if st.session_state.analysis_warnings:
                for warning in st.session_state.analysis_warnings:
                    st.error(warning)
            else:
                st.success("No collision events were detected.")
            st.markdown("</div>", unsafe_allow_html=True)

            with open(st.session_state.analysis_report_path, "rb") as pdf_file:
                st.download_button("Download PDF report", pdf_file, file_name="AnatoScope_Report.pdf", mime="application/pdf", key="download_pdf_report")

with main_tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📹 Real-Time Live Feed / Endoscope Inspection")
    st.caption("Capture a frame from your endoscope camera for instantaneous AI detection and tool tracking.")
    
    img_file_buffer = st.camera_input("Take a snapshot from live feed")

    if img_file_buffer is not None:
        # تحويل الصورة الملتقطة إلى صيغة OpenCV
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("Analyzing live snapshot..."):
            # إجراء الفحص التفاعلي السريع
            tools_res = tools_model(cv2_img, conf=confidence)[0]
            anatomy_res = anatomy_model(cv2_img, conf=confidence)[0]
            
            annotated_frame = cv2_img.copy()
            frame_shape = annotated_frame.shape[:2]

            for mask in iter_mask_candidates(anatomy_res.masks):
                mask_arr = resize_mask_to_frame(mask, frame_shape)
                if mask_arr is not None:
                    annotated_frame[mask_arr] = [0, 255, 0]

            if tools_res.boxes is not None:
                annotated_frame = tools_res.plot(img=annotated_frame, conf=False)
            
            # تحويل الألوان للـ Streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            col_live1, col_live2 = st.columns(2)
            with col_live1:
                st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption="Original Frame", use_container_width=True)
            with col_live2:
                st.image(annotated_frame_rgb, caption="AI Annotated Frame", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
