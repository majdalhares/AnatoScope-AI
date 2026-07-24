
import os
import cv2
import torch
import numpy as np
import pandas as pd
import gradio as gr
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_half = True if device == 'cuda' else False

anatomy_model = YOLO("weights/anatomy_best.pt").to(device)
tools_model = YOLO("weights/tools_best.pt").to(device)

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
    story.append(Paragraph("<b>Surgical AI Video Analysis Report</b>", title_style))
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
        story.append(Paragraph("No critical collisions or hazards detected.", styles['Normal']))

    doc.build(story)
    return pdf_path

def process_surgical_video_advanced(video_file, conf_threshold, frame_skip, progress=gr.Progress(track_tqdm=True)):
    if video_file is None:
        return None, "⚠️ يرجى رفع ملف فيديو جراحي أولاً.", None, None

    video_path = video_file.name if hasattr(video_file, 'name') else str(video_file)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    if width == 0 or height == 0:
        return None, "⚠️ تعذر قراءة أبعاد الفيديو.", None, None

    target_width = 480
    target_height = int(height * (target_width / width))

    output_path = "advanced_surgical_analysis.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    detected_summary = {}
    warnings_log = []
    unique_warnings = set()
    current_frame = 0

    progress(0, desc="🚀 بدء المعالجة...")

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1

            if total_frames > 0 and current_frame % 5 == 0:
                percent = current_frame / total_frames
                progress(percent, desc=f"⚡ تحليل: {int(percent*100)}%")

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
                            msg = f"🚨 تلامس خطير [{time_sec}s]: الأداة #{tool_id} ({tool_name}) تُلامس ({organ_name})"
                            if msg not in unique_warnings:
                                unique_warnings.add(msg)
                                warnings_log.append(msg)

            out.write(combined_frame)

    cap.release()
    out.release()

    pdf_file = generate_pdf_report(detected_summary, warnings_log, total_frames, fps)
    df_data = [{"العنصر / الفئة": k, "تكرار الظهور": v} for k, v in detected_summary.items()]
    df_summary = pd.DataFrame(df_data) if df_data else pd.DataFrame(columns=["العنصر", "تكرار الظهور"])
    warnings_text = "
".join(warnings_log) if warnings_log else "✅ البيئة الجراحية آمنة."

    return output_path, warnings_text, df_summary, pdf_file

with gr.Blocks(title="المنظومة الجراحية الذكية") as demo:
    gr.Markdown("# 🏥 المنظومة الجراحية المتقدمة (Surgical AI Pipeline)")
    with gr.Row():
        with gr.Column():
            video_input = gr.File(label="📁 رفع ملف الفيديو الجراحي", file_types=[".mp4", ".avi", ".mov"])
            conf_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.30, step=0.05, label="🎯 مستوى الثقة")
            skip_slider = gr.Slider(minimum=1, maximum=5, value=4, step=1, label="🚀 معدل التسريع")
            btn_analyze = gr.Button("🚀 بدء التحليل الجراحي", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="🎬 الفيديو المحلل")
            warnings_output = gr.Textbox(label="🚨 سجل التحذيرات والتلامس", lines=5)
            pdf_output = gr.File(label="📄 تحميل التقرير (PDF)")
    with gr.Row():
        table_output = gr.Dataframe(label="📊 ملخص العناصر")

    btn_analyze.click(fn=process_surgical_video_advanced, inputs=[video_input, conf_slider, skip_slider], outputs=[video_output, warnings_output, table_output, pdf_output])

if __name__ == "__main__":
    demo.launch()
