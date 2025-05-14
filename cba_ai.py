import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import getpass
from pathlib import Path
from datetime import datetime

# Prepare directory structure
user_name = getpass.getuser()
base_dir = Path(f"C:/Users/{user_name}/Downloads/results")
images_dir = base_dir / "images"
videos_dir = base_dir / "videos"

# Create directories if they don't exist
base_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(exist_ok=True)
videos_dir.mkdir(exist_ok=True)

# Load both YOLO models globally
try:
    gun_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')  # Gun detection model
    banner_model = YOLO('./runs/detect/train/weights/best.pt')  # Banner detection model
    st.success("Dono YOLO models successfully load ho gaye!")
except Exception as e:
    st.error(f"Error loading YOLO models: {e}")
    st.stop()

# Custom function to draw bounding boxes and labels from both models
def draw_combined_results(frame, gun_results, banner_results):
    combined_frame = frame.copy()
    
    # Gun model ke results draw karo (Green color mein)
    if gun_results and len(gun_results) > 0:
        gun_result = gun_results[0]  # First result
        if gun_result.boxes:
            for box in gun_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{gun_model.names[cls]} {conf:.2f}"
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(combined_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Banner model ke results draw karo (Red color mein)
    if banner_results and len(banner_results) > 0:
        banner_result = banner_results[0]  # First result
        if banner_result.boxes:
            for box in banner_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{banner_model.names[cls]} {conf:.2f}"
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(combined_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return combined_frame

def detect_image(image_file):
    try:
        st.write("Image process ho rahi hai...")
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error("Error: Image read nahi ho saki")
            return None, None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"detected_{timestamp}.jpg"
        output_path = images_dir / output_filename
        
        # Dono models ko image pe run karo
        gun_results = gun_model(image, verbose=False)
        banner_results = banner_model(image, verbose=False)
        
        # Dono ke results combine karo
        annotated_img = draw_combined_results(image, gun_results, banner_results)
        
        # Annotated image save karo
        cv2.imwrite(str(output_path), annotated_img)
        
        st.write(f"Image process ho gayi aur save ho gayi: {output_path}")
        return annotated_img, str(output_path)
    except Exception as e:
        st.error(f"Error image process karte waqt: {e}")
        return None, None

def detect_video(video_file):
    try:
        st.write("Video process start ho rahi hai...")
        
        # Temporary file create karo
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_data = video_file.read()
        tfile.write(video_data)
        tfile.close()
        
        if not os.path.exists(tfile.name) or os.path.getsize(tfile.name) == 0:
            st.error("Error: Temporary video file invalid ya khali hai")
            return None
        
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error(f"Error: Video {video_file.name} open nahi ho saki")
            os.unlink(tfile.name)
            return None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.total_frames = total_frames
        st.write(f"Video mein total frames: {total_frames}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"detected_{timestamp}.mp4"
        output_path = videos_dir / output_filename
        
        # H.264 codec use karo for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            st.error("Error: Video writer initialize nahi ho saka")
            cap.release()
            os.unlink(tfile.name)
            return None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_count = 0
        st.session_state.frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Dono models ko frame pe run karo
                gun_results = gun_model(frame, verbose=False)
                banner_results = banner_model(frame, verbose=False)
                
                # Dono ke results combine karo
                annotated_frame = draw_combined_results(frame, gun_results, banner_results)
                
                # Annotated frame ko output video mein write karo
                out.write(annotated_frame)
                frame_count += 1
                st.session_state.frame_count = frame_count
                
                if frame_count % 10 == 0:
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
                
                if frame_count % 100 == 0:
                    st.write(f"Processed {frame_count} frames")
                    
            except Exception as e:
                st.write(f"Error frame {frame_count} process karte waqt: {e}")
                continue
        
        # Resources release karo
        cap.release()
        out.release()
        os.unlink(tfile.name)
        
        # Output file verify karo
        if os.path.exists(output_path):
            cap_test = cv2.VideoCapture(str(output_path))
            if cap_test.isOpened():
                st.write(f"Video validated: {cap_test.get(cv2.CAP_PROP_FRAME_COUNT)} frames")
                cap_test.release()
            else:
                st.error("Output video corrupt hai")
                return None
        else:
            st.error("Output video create nahi hui")
            return None
        
        progress_bar.progress(1.0)
        status_text.text(f"Completed! Processed {frame_count}/{total_frames} frames")
        st.success(f"Video process ho gayi. Output save ho gaya: {output_path}")
        
        return str(output_path.resolve())
    except Exception as e:
        st.error(f"Error video process karte waqt: {e}")
        return None

def detect_live_camera():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Camera access nahi ho saka")
            return
        
        stframe = st.empty()
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera se frame capture nahi ho saka")
                break
            
            try:
                # Dono models ko live frame pe run karo
                gun_results = gun_model(frame, verbose=False)
                banner_results = banner_model(frame, verbose=False)
                
                # Dono ke results combine karo
                annotated_frame = draw_combined_results(frame, gun_results, banner_results)
                
                # RGB mein convert karo for Streamlit display
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame, channels="RGB", use_container_width=True)
                
            except Exception as e:
                st.error(f"Error live frame process karte waqt: {e}")
                break
        
        cap.release()
        stframe.empty()
    except Exception as e:
        st.error(f"Error live detection mein: {e}")

# Streamlit app
st.title("Crowd Behaviour Analysis System")
st.subheader("Weapon Placards & Behaviour Detection using YOLOv8")
st.write("Yeh app YOLOv8 models use karta hai taake images, videos, aur live camera feed mein weapons aur banners detect kar sake.")
st.write("Image ya video upload karo taake YOLO model se weapons aur banners detect ho saken.")

# Initialize session state variables
if 'image_processed' not in st.session_state:
    st.session_state.image_processed = False
    st.session_state.image_path = None
    st.session_state.annotated_img = None
    st.session_state.image_file = None

if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
    st.session_state.video_path = None
    st.session_state.frame_count = 0
    st.session_state.total_frames = 0

if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Real Time Detection"])

with tab1:
    st.header("Image Detection")
    
    if not st.session_state.image_processed:
        image_file = st.file_uploader("Image Upload Karo", type=["jpg", "jpeg", "png"], key="image_uploader")
        if image_file is not None:
            st.session_state.image_file = image_file
            annotated_img, output_path = detect_image(image_file)
            if annotated_img is not None:
                st.session_state.annotated_img = annotated_img
                st.session_state.image_path = output_path
                st.session_state.image_processed = True
                st.rerun()
    
    if st.session_state.image_processed and st.session_state.annotated_img is not None:
        annotated_img_rgb = cv2.cvtColor(st.session_state.annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_img_rgb, caption="Detected Image", use_container_width=True)
        
        with open(st.session_state.image_path, "rb") as file:
            btn = st.download_button(
                label="Detected Image Download Karo",
                data=file,
                file_name=os.path.basename(st.session_state.image_path),
                mime="image/jpeg",
                key="image_download_button"
            )
            if btn:
                st.success("Download start ho gaya!")
        
        if st.button("Nayi Image Process Karo"):
            st.session_state.image_processed = False
            st.session_state.image_path = None
            st.session_state.annotated_img = None
            st.session_state.image_file = None
            st.rerun()

with tab2:
    st.header("Video Detection")
    
    if not st.session_state.video_processed:
        video_file = st.file_uploader("Video Upload Karo", type=["mp4", "avi", "mov"], key="video_uploader")
        if video_file is not None:
            output_path = detect_video(video_file)
            if output_path:
                st.session_state.video_path = output_path
                st.session_state.video_processed = True
                st.rerun()
    
    if st.session_state.video_processed and st.session_state.video_path:
        try:
            with open(st.session_state.video_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.write(f"Total frames processed: {st.session_state.frame_count}/{st.session_state.total_frames}")
            st.write(f"Video file size: {os.path.getsize(st.session_state.video_path) / (1024*1024):.2f} MB")
            st.download_button(
                label="Detected Video Download Karo",
                data=video_bytes,
                file_name=os.path.basename(st.session_state.video_path),
                mime="video/mp4"
            )
        except Exception as e:
            st.error(f"Error video display karte waqt: {e}")
            st.write("Download kar ke dekho")
        
        if st.button("Nayi Video Process Karo"):
            st.session_state.video_processed = False
            st.session_state.video_path = None
            st.session_state.frame_count = 0
            st.session_state.total_frames = 0
            st.rerun()

with tab3:
    st.header("Live Camera Detection")
    start_camera = st.button("Live Detection Start Karo")
    stop_camera = st.button("Live Detection Stop Karo")

    if start_camera:
        st.session_state.camera_active = True
        detect_live_camera()
    if stop_camera:
        st.session_state.camera_active = False