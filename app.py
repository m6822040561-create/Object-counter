import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
import tempfile
import subprocess

# --- 2. ฟังก์ชันคำนวณ IoU ---
def get_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / float(area1 + area2 - intersection + 1e-6)

# --- 3. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Recycle Conveyor Counter", layout="wide")
st.title("♻️ ระบบตรวจจับและคัดแยกขยะบนสายพาน")

# --- 4. โหลดโมเดล ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 5. Sidebar ตั้งค่า ---
st.sidebar.header("การตั้งค่าโมเดล")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.2)
iou_threshold = st.sidebar.slider("NMS IoU Threshold", 0.1, 1.0, 0.6)
overlap_check = st.sidebar.slider("Overlap Check (สำหรับการนับ)", 0.05, 0.5, 0.1)

class_names = {0: "can", 1: "glass", 2: "paperpack", 3: "plastic"}

# --- 6. อัปโหลดและประมวลผลวิดีโอ ---
uploaded_video = st.file_uploader("อัปโหลดวิดีโอสายพาน (mp4, mov)...", type=['mp4', 'mov'])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) # ดึงค่า FPS เพื่อให้วิดีโอเล่นเนียนเท่าต้นฉบับ
    if fps == 0 or fps is None: fps = 30
    
    # 1. เตรียมไฟล์สำหรับแอบบันทึกวิดีโอเบื้องหลัง
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_out.name, fourcc, int(fps), (640, 360))
    
    col1, col2 = st.columns([3, 1])
    st_frame = col1.empty()
    st_stats = col2.empty()
    st_progress = st.empty()
    
    prev_boxes = []
    counts = {name: 0 for name in class_names.values()}
    total_count = 0
    frame_count = 0
    
    st_progress.info("⏳ กำลังเตรียมวิดีโอและเริ่มประมวลผล กรุณารอสักครู่...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        frame = cv2.resize(frame, (640, 360))
        w, h = 640, 360
        
        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        raw_boxes = results.boxes.xyxy.cpu().numpy()
        raw_cls = results.boxes.cls.cpu().numpy().astype(int)
        
        current_boxes = []
        current_cls = []
        
        for i, box in enumerate(raw_boxes):
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < (w * 0.8) and bh < (h * 0.8):
                current_boxes.append(box)
                current_cls.append(raw_cls[i])
                
        for i, c_box in enumerate(current_boxes):
            cls_id = current_cls[i]
            label = class_names.get(cls_id, f"id_{cls_id}")
            
            x1, y1, x2, y2 = map(int, c_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, label.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            is_new = True
            for p_box in prev_boxes:
                if get_iou(c_box, p_box) > overlap_check:
                    is_new = False
                    break
            if is_new:
                if label in counts: counts[label] += 1
                total_count += 1
                
        prev_boxes = current_boxes
        
        y_pos = 50
        for name, val in counts.items():
            cv2.putText(frame, f'{name.upper()}: {val}', (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_pos += 30
        cv2.putText(frame, f'TOTAL: {total_count}', (30, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # 2. บันทึกภาพลงไฟล์วิดีโอทุกเฟรม
        out.write(frame)
        
        # แสดงผลโชว์ความคืบหน้าบนเว็บ
        if frame_count % 3 == 0:
            st_progress.info(f"⚙️ กำลังประมวลผล... (ผ่านไปแล้ว {frame_count} เฟรม)")
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            with st_stats.container():
                st.markdown("### 📊 สถิติแบบ Real-time")
                st.metric("Total Objects", total_count)
                for name, val in counts.items():
                    st.metric(f"🟢 {name.capitalize()}", val)

    cap.release()
    out.release() 
    st_frame.empty() # ล้างภาพนิ่งที่ค้างหน้าจอออก
    
    # 3. แปลงไฟล์วิดีโอให้รองรับบนเว็บไซต์
    st_progress.info("🔄 ประมวลผลเสร็จแล้ว กำลังสร้างเครื่องเล่นวิดีโอ...")
    final_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    subprocess.run(["ffmpeg", "-y", "-i", temp_out.name, "-vcodec", "libx264", "-preset", "ultrafast", final_out.name], capture_output=True)
    
    st_progress.success(f"✅ เสร็จสมบูรณ์! ตรวจจับวัตถุไปทั้งหมด {total_count} ชิ้น")
    
    # 4. แสดงเครื่องเล่นวิดีโอของแท้ (กด Play/Pause/เลื่อนเวลาได้)
    st.video(final_out.name)
# ... (โค้ดส่วนประมวลผลและแปลงไฟล์ด้วย ffmpeg เดิม) ...
    
    st_progress.success(f"✅ เสร็จสมบูรณ์! ตรวจจับวัตถุไปทั้งหมด {total_count} ชิ้น")
    
    # 1. แสดงเครื่องเล่นวิดีโอให้ดูบนเว็บ
    st.video(final_out.name)
    
    # 2. เพิ่มปุ่มดาวน์โหลดไฟล์ลงเครื่อง
    with open(final_out.name, "rb") as file:
        btn = st.download_button(
            label="💾 ดาวน์โหลดวิดีโอผลลัพธ์",
            data=file,
            file_name="detected_conveyor_video.mp4",
            mime="video/mp4"
        )
