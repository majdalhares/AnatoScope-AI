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

# إعدادات الصفحة
st.set_page_config(
    page_title="AnatoScope AI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AnatoScope AI")
st.caption("Real-Time Surgical Safety & Anatomical Vision System")
st.markdown("---")

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
    story.append(Paragraph("<b>AnatoScope AI - Surgical Report</b>", title_style))
    story.append(Spacer(1, 12))

    duration_sec = round(total_frames / (fps if fps > 0 else 25), 2)
    story.append(Paragraph(f"<b>Total Video Duration:</b> {duration_sec} seconds ({total_frames} frames)", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>1. Detected Anatomy & Tools Summary:</b>", styles['Heading2']))
    table_data = [["Element / Class", "Frames Detected", "Presence Ratio"]]
    for item, count in summary_dict.items():
        ratio = f"{round((count / total_frames) * 100, 1)}%" if total_frames > 0 else "0%"
        table_data.append([str(item), str(count), ratio])
    
    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>2. Safety Hazards & Contact Warnings Log:</b>", styles['Heading2']))
    if warnings_list:
        for w in warnings_list:
            story.append(Paragraph(f"• {w}", styles['Normal']))
    else:
        story.append(Paragraph("No critical collisions detected.", styles['Normal']))

    doc.build(story)
    return pdf_path

# الشريط الجانبي للإعدادات
with st.sidebar:
    st.header("⚙️ إعدادات التحليل")
    conf_threshold = st.slider("🎯 مستوى الثقة (Confidence)", 0.1, 0.9, 0.30, 0.05)
    frame_skip = st.slider("🚀 معدل التسريع (Frame Skip)", 1, 5, 4)

uploaded_file = st.file_uploader("📁 رفع فيديو جراحي للتحليل", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    if st.button("🚀 بدء التحليل الجراحي الشامل", type="primary"):
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
                    status_text.text(f"⏳ معالجة الإطار {current_frame} / {total_frames}")

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
                                msg = f"🚨 collision [{time_sec}s]: Tool #{tool_id} ({tool_name}) touched ({organ_name})"
                                if msg not in unique_warnings:
                                    unique_warnings.add(msg)
                                    warnings_log.append(msg)

                out.write(combined_frame)

        cap.release()
        out.release()
        status_text.text("✅ انتهى التحليل!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎬 الفيديو المحلل")
            st.video(output_path)
        
        with col2:
            st.subheader("🚨 سجل التحذيرات والتلامس")
            if warnings_log:
                for w in warnings_log:
                    st.error(w)
            else:
                st.success("✅ البيئة الجراحية آمنة")

            pdf_file = generate_pdf_report(detected_summary, warnings_log, total_frames, fps)
            with open(pdf_file, "rb") as f:
                st.download_button("📄 تحميل التقرير الطبي (PDF)", f, file_name="AnatoScope_Report.pdf", mime="application/pdf")

        st.subheader("📊 ملخص الكشف")
        df_data = [{"العنصر": k, "تكرار الظهور": v} for k, v in detected_summary.items()]
        st.dataframe(pd.DataFrame(df_data))
