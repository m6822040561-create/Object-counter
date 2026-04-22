import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# --- 1. ฟังก์ชันคำนวณ IoU (จาก Colab ของคุณ) ---
def get_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / float(area1 + area2 - intersection + 1e-6)

# --- 2. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Recycle Conveyor Counter", layout="wide")
st.title("♻️ ระบบตรวจจับและคัดแยกขยะบนสายพาน")

# --- 3. โหลดโมเดล ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 4. Sidebar ตั้งค่า (ดึงค่ามาจาก Cell 5 ของคุณ) ---
st.sidebar.header("การตั้งค่าโมเดล")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.2)
iou_threshold = st.sidebar.slider("NMS IoU Threshold", 0.1, 1.0, 0.6)
overlap_check = st.sidebar.slider("Overlap Check (สำหรับการนับ)", 0.05, 0.5, 0.1)

class_names = {0: "can", 1: "glass", 2: "paperpack", 3: "plastic"}

# --- 5. อัปโหลดและประมวลผลวิดีโอ ---
uploaded_video = st.file_uploader("อัปโหลดวิดีโอสายพาน (mp4, mov)...", type=['mp4', 'mov'])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # แบ่งหน้าจอแสดงผล: ซ้ายวิดีโอ ขวาสถิติ
    col1, col2 = st.columns([3, 1])
    st_frame = col1.empty()
    st_stats = col2.empty()
    
    prev_boxes = []
    counts = {name: 0 for name in class_names.values()}
    total_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # รันโมเดล
        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        raw_boxes = results.boxes.xyxy.cpu().numpy()
        raw_cls = results.boxes.cls.cpu().numpy().astype(int)
        
        current_boxes = []
        current_cls = []
        
        # กรอง Noise
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
            
            # ลอจิกการนับของคุณ
            is_new = True
            for p_box in prev_boxes:
                if get_iou(c_box, p_box) > overlap_check:
                    is_new = False
                    break
            if is_new:
                if label in counts: counts[label] += 1
                total_count += 1
                
        prev_boxes = current_boxes
        
        # พิมพ์ตัวเลขลงบนวิดีโอ (เหมือน Colab)
        y_pos = 100
        for name, val in counts.items():
            cv2.putText(frame, f'{name.upper()}: {val}', (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            y_pos += 50
        cv2.putText(frame, f'TOTAL: {total_count}', (50, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
        
        # อัปเดตวิดีโอบน Streamlit (แปลง BGR เป็น RGB)
        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # อัปเดตตารางสถิติด้านขวา
        with st_stats.container():
            st.markdown("### 📊 สถิติแบบ Real-time")
            st.metric("Total Objects", total_count)
            for name, val in counts.items():
                st.metric(f"🟢 {name.capitalize()}", val)

    cap.release()
    st.success("✅ ประมวลผลวิดีโอเสร็จสิ้น!")
