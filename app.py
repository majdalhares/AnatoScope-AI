import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import tempfile
import os
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ---------------------------------------------------------
# 1. إعدادات الصفحة والهوية البصرية Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="AnatoScope AI | Surgical Vision",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تصميم عصري مخصص بأسلوب المستشفيات الحديثة والـ Dashboards
st.markdown("""
<style>
    /* خلفية الصفحة الرئيسية */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* الهيدر والعناوين */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* بطاقات الإحصائيات Top Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* صندوق التحذيرات */
    .hazard-box {
        background-color: rgba(239, 68, 68, 0.1);
        border-right: 4px solid #ef4444;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }

    .safe-box {
        background-color: rgba(34, 197, 94, 0.1);
        border-right: 4px solid #22c55e;
        padding: 15px;
        border-radius: 6px;
        text-align: center;
        font-weight: bold;
        color: #4ade80;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. تحميل النماذج (Models Loading)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    anatomy_model = YOLO("anatomy_best.pt").to(device)
    tools_model = YOLO("tools_best.pt").to(device)
    return anatomy_model, tools_model, device

anatomy_model, tools_model, device = load_models()
use_half = True if device == 'cuda' else False

def check_mask_collision(mask1, mask2, target_shape):
    if mask1 is None or mask2 is None:
        return False
    m1_resized = cv2.resize(mask1.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    m2_resized = cv2.resize(mask2.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.any(np.logical_and(m1_resized > 0, m2_resized > 0))

def generate_pdf_report(summary_dict, warnings_list, total_frames, fps):
    pdf_path = "Surgical_Analysis_Report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, alignment=1)
    story.append(Paragraph("<b>AnatoScope AI - Surgical Safety Report</b>", title_style))
    story.append(Spacer(1, 15))

    duration_sec = round(total_frames / (fps if fps > 0 else 25), 2)
    story.append(Paragraph(f"<b>Video Duration:</b> {duration_sec} s | <b>Processed Frames:</b> {total_frames}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>1. Detections Breakdown:</b>", styles['Heading2']))
    table_data = [["Class / Element", "Frames Present", "Presence Ratio"]]
    for item, count in summary_dict.items():
        ratio = f"{round((count / total_frames) * 100, 1)}%" if total_frames > 0 else "0%"
        table_data.append([str(item), str(count), ratio])
    
    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>2. Critical Contact Events Log:</b>", styles['Heading2']))
    if warnings_list:
        for w in warnings_list:
            story.append(Paragraph(f"• {w}", styles['Normal']))
    else:
        story.append(Paragraph("Zero critical organ collisions detected.", styles['Normal']))

    doc.build(story)
    return pdf_path

# ---------------------------------------------------------
# 3. الهيدر والشريط الجانبي
# ---------------------------------------------------------
st.markdown('<div class="main-title">🩺 AnatoScope AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Next-Generation Surgical Computer Vision & Real-Time Hazard Intelligence</div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=70)
    st.header("⚙️ Control Panel")
    conf_threshold = st.slider("🎯 AI Confidence Threshold", 0.1, 0.9, 0.30, 0.05)
    frame_skip = st.slider("⚡ Processing Acceleration (Skip)", 1, 5, 3)
    st.markdown("---")
    st.info("💡 **Tip for Judges:** Lower confidence detects subtle tools, while higher confidence reduces false positives.")

# ---------------------------------------------------------
# 4. جسم التطبيق ورفع الملفات
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📂 Drag & Drop Surgical Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    if st.button("🚀 Run Intelligent Surgical Analysis", type="primary", use_container_width=True):
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        target_width = 480
        target_height = int(height * (target_width / width)) if width > 0 else 360

        output_path = "output_surgical.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

        progress_bar = st.progress(0)
        status_text = st.empty()

        detected_summary = {}
        warnings_log = []
        unique_warnings = set()
        current_frame = 0

        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame += 1

                if total_frames > 0:
                    progress_bar.progress(current_frame / total_frames)
                    status_text.markdown(f"⏳ **Analyzing Frame:** `{current_frame}/{total_frames}`")

                if current_frame % frame_skip != 0:
                    continue

                resized_frame = cv2.resize(frame, (target_width, target_height))
                tools_res = tools_model.track(resized_frame, conf=conf_threshold, imgsz=320, half=use_half, persist=True, verbose=False)[0]
                anatomy_res = anatomy_model(resized_frame, conf=conf_threshold, imgsz=320, half=use_half, verbose=False)[0]

                overlay = resized_frame.copy()
                if anatomy_res.masks is not None:
                    for mask in anatomy_res.masks.data:
                        m = mask.cpu().numpy().astype(np.uint8)
                        m_resized = cv2.resize(m, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                        overlay[m_resized == 1] = overlay[m_resized == 1] * 0.7 + np.array([0, 255, 0]) * 0.3

                combined_frame = cv2.addWeighted(overlay, 0.8, resized_frame, 0.2, 0)
                if tools_res.boxes is not None:
                    combined_frame = tools_res.plot(img=combined_frame, conf=False)

                anatomy_classes = [anatomy_model.names[int(c)] for c in anatomy_res.boxes.cls] if anatomy_res.boxes is not None else []
                tools_classes = [tools_model.names[int(c)] for c in tools_res.boxes.cls] if tools_res.boxes is not None else []

                for item in anatomy_classes + tools_classes:
                    detected_summary[item] = detected_summary.get(item, 0) + 1

                if tools_res.masks is not None and anatomy_res.masks is not None:
                    for i, t_mask in enumerate(tools_res.masks.data):
                        tool_name = tools_model.names[int(tools_res.boxes.cls[i])] if tools_res.boxes is not None else "Tool"
                        tool_id = int(tools_res.boxes.id[i]) if (tools_res.boxes is not None and tools_res.boxes.id is not None) else i
                        for j, a_mask in enumerate(anatomy_res.masks.data):
                            organ_name = anatomy_model.names[int(anatomy_res.boxes.cls[j])] if anatomy_res.boxes is not None else "Organ"
                            if check_mask_collision(t_mask.cpu().numpy(), a_mask.cpu().numpy(), (target_height, target_width)):
                                time_sec = round(current_frame / fps, 1)
                                msg = f"🚨 Collision [{time_sec}s]: Tool #{tool_id} ({tool_name}) touched ({organ_name})"
                                if msg not in unique_warnings:
                                    unique_warnings.add(msg)
                                    warnings_log.append(msg)

                out.write(combined_frame)

        cap.release()
        out.release()
        status_text.markdown("✅ **Analysis Complete!**")

        # ---------------------------------------------------------
        # 5. عرض النتائج والمؤشرات التفاعلية (KPIs)
        # ---------------------------------------------------------
        st.markdown("---")
        
        # كروت الأرقام المباشرة للجنة التحكيم والأطباء
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-val">{total_frames}</div><div class="metric-lbl">Total Video Frames</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-val">{len(detected_summary)}</div><div class="metric-lbl">Unique Detected Classes</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-val">{len(warnings_log)}</div><div class="metric-lbl">Safety Hazards Logged</div></div>', unsafe_allow_html=True)
        
        safety_score = max(0, 100 - (len(warnings_log) * 15))
        m4.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{"#22c55e" if safety_score > 70 else "#ef4444"};">{safety_score}%</div><div class="metric-lbl">Surgical Safety Score</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("🎬 Visual Tracking Output")
            st.video(output_path)
        
        with col2:
            st.subheader("🚨 Real-Time Risk & Hazard Monitor")
            if warnings_log:
                for w in warnings_log:
                    st.markdown(f'<div class="hazard-box">{w}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="safe-box">🛡️ Optimal Surgical Safety: No hazardous organ collisions detected.</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            pdf_file = generate_pdf_report(detected_summary, warnings_log, total_frames, fps)
            with open(pdf_file, "rb") as f:
                st.download_button("📄 Download Clinical PDF Report", f, file_name="AnatoScope_Surgical_Report.pdf", mime="application/pdf", use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Quantitative Object Frequency Data")
        if detected_summary:
            df_data = pd.DataFrame([{"Class Name": k, "Frame Occurrences": v} for k, v in detected_summary.items()])
            st.dataframe(df_data, use_container_width=True)
