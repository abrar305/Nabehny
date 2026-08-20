import streamlit as st
import cv2
import numpy as np
import os
import sqlite3
from database import add_student, add_student_image, get_connection
from core.face_engine import FaceEngine

@st.cache_resource
def load_engine():
    return FaceEngine()

def apply_custom_css():
    st.markdown("""
        <style>
        .main-card {
            background-color: #1E222D;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-bottom: 20px;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .status-box {
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

def render_add_student_page():
    apply_custom_css()
    face_engine = load_engine()
    
    # Initialize Session State
    if "captured_images" not in st.session_state:
        st.session_state.captured_images = []

    # Title Banner
    st.title("🎓 Student Registration Portal")
    st.caption("Enroll new students by uploading photos or capturing them via webcam for face recognition.")
    st.markdown("---")

    # Layout using Columns
    col_info, col_input = st.columns([1, 2], gap="large")

    with col_info:
        st.subheader("📌 Registration Steps")
        st.markdown("""
        1. **Enter Full Name**: Make sure to use the official name.
        2. **Provide 5 Photos**: 
           - Good lighting condition.
           - Clear face visibility.
           - Vary the head angle slightly for best accuracy.
        3. **Save**: The system will extract embeddings and update the AI model automatically.
        """)

    with col_input:
        student_name = st.text_input("👤 Full Student Name:", placeholder="e.g., Ahmed Ali")

        input_method = st.radio(
            "📷 Image Input Method:",
            ["Upload 5 Images", "Capture via Webcam"],
            horizontal=True
        )

        images_to_process = []

        # --- Option A: Upload Files ---
        if input_method == "Upload 5 Images":
            st.session_state.captured_images = []  # Reset camera state
            uploaded_files = st.file_uploader(
                "Choose exactly 5 student photos", 
                type=['jpg', 'jpeg', 'png'], 
                accept_multiple_files=True
            )
            if uploaded_files:
                if len(uploaded_files) != 5:
                    st.warning(f"⚠️ Please select exactly 5 images. Current selection: **{len(uploaded_files)}**")
                else:
                    for file in uploaded_files:
                        bytes_data = file.read()
                        cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                        images_to_process.append((file.name, cv_img))

        # --- Option B: Webcam Capture ---
        elif input_method == "Capture via Webcam":
            current_count = len(st.session_state.captured_images)
            
            # Progress status bar
            progress = current_count / 5.0
            st.progress(progress)
            st.write(f"📸 **Captured Photos:** `{current_count} / 5`")

            if current_count < 5:
                picture = st.camera_input("Position face and click 'Take Photo'", key=f"cam_input_{current_count}")
                if picture:
                    bytes_data = picture.read()
                    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    st.session_state.captured_images.append(cv_img)
                    st.rerun()
            else:
                st.success("✅ All 5 photos captured successfully!")

            if current_count > 0:
                if st.button("🔄 Reset / Recapture Photos", use_container_width=True):
                    st.session_state.captured_images = []
                    st.rerun()

            if st.session_state.captured_images:
                st.write("##### Photo Preview:")
                cols = st.columns(5)
                for idx, img in enumerate(st.session_state.captured_images):
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cols[idx].image(rgb_img, caption=f"#{idx+1}", use_container_width=True)

            if len(st.session_state.captured_images) == 5:
                images_to_process = [(f"cam_img_{i+1}.jpg", img) for i, img in enumerate(st.session_state.captured_images)]

    st.markdown("---")

    # --- Save Button ---
    if st.button("💾 Complete Registration & Save", type="primary", use_container_width=True):
        if not student_name.strip():
            st.error("❌ Please enter the student's full name first.")
            return
            
        if len(images_to_process) != 5:
            st.error("❌ Exactly 5 images are required to complete registration.")
            return

        # Insert student record
        stu_id = add_student(student_name)
        if not stu_id:
            st.error("❌ Database Error: Failed to add student.")
            return

        student_dir = os.path.join("data", "student_images", f"stu_{stu_id}")
        os.makedirs(student_dir, exist_ok=True)
        
        saved_count = 0
        status_container = st.container()

        with st.spinner("⚡ Extracting face embeddings and saving dataset..."):
            for idx, (filename, img) in enumerate(images_to_process):
                embedding, cropped_face = face_engine.get_face_embedding(img)
                
                if embedding is None:
                    status_container.error(f"⚠️ Photo #{idx+1}: No face detected — Skipped.")
                    continue
                
                save_path = os.path.join(student_dir, f"img_{idx+1}.jpg")
                cv2.imwrite(save_path, img)
                
                # Save each embedding individually into database
                if add_student_image(stu_id=stu_id, image_path=save_path, embedding_array=embedding):
                    saved_count += 1
                    status_container.success(f"✅ Photo #{idx+1}: Face registered successfully.")

        if saved_count > 0:
            st.balloons()
            st.success(f"🎉 **Student '{student_name}' registered successfully with {saved_count} face embeddings!**")
            st.session_state.captured_images = []
        else:
            # Clean up empty student entry if no faces were saved
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM Students WHERE Stu_ID = ?", (stu_id,))
            conn.commit()
            conn.close()
            st.error("❌ Failed to register student. No valid faces were detected in the provided images.")