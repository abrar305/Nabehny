import os
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from database import add_subject, add_class, get_connection
from gui.add_student import render_add_student_page
from gui.dashboard import render_dashboard_page
from core.video_processor import VideoProcessor

def apply_global_styles():
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
        }
        .css-1d3b13e {
            background-color: #161B22;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .status-card {
            background-color: #1E222D;
            border-radius: 10px;
            padding: 16px;
            border-left: 4px solid #10B981;
            margin-top: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

def run_app():
    st.set_page_config(
        page_title="Smart Classroom System",
        page_icon="🎓",
        layout="wide"
    )

    apply_global_styles()

    with st.sidebar:
        st.title("🎓 Smart Classroom")
        st.caption("AI-Powered Attendance & Behavior System")
        st.markdown("---")
        
        selected = option_menu(
            menu_title="Main Navigation",
            options=["Register Student", "Process Media", "Analytics Dashboard", "Manage Subjects & Classes"],
            icons=["person-plus-fill", "camera-reels-fill", "bar-chart-fill", "journal-bookmark-fill"],
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "transparent"},
                "icon": {"color": "#3B82F6", "font-size": "18px"}, 
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "2px", "--hover-color": "#21262D"},
                "nav-link-selected": {"background-color": "#238636", "font-weight": "600"},
            }
        )

    # --- Page 1: Student Registration ---
    if selected == "Register Student":
        render_add_student_page()

    # --- Page 2: Lecture Processing (Video or Image) ---
    elif selected == "Process Media":
        st.title("📹 Upload & Process Lecture Media")
        st.caption("Process live classroom recordings or snapshots to detect student presence and behavioral events.")
        
        conn = get_connection()
        classes_df = pd.read_sql_query("""
            SELECT c.Class_ID, s.Name as Subject_Name 
            FROM Classes c 
            JOIN Subjects s ON c.Sub_ID = s.Sub_ID
        """, conn)
        conn.close()

        if classes_df.empty:
            st.error("⚠️ No classes found. Please create a subject and class session first under 'Manage Subjects & Classes'.")
        else:
            class_options = {f"Class #{row['Class_ID']} — {row['Subject_Name']}": row['Class_ID'] 
                             for _, row in classes_df.iterrows()}

            col_select, col_type = st.columns([2, 1], gap="medium")
            
            with col_select:
                selected_class_name = st.selectbox(
                    "📚 Select Target Class Session:", 
                    options=list(class_options.keys()),
                    key="target_class_select"
                )
                selected_class_id = int(class_options[selected_class_name])

            with col_type:
                media_type = st.radio(
                    "📸 Select Input Media Type:", 
                    ["Video File", "Classroom Image"], 
                    horizontal=True
                )

            st.markdown("---")

            # --- Option A: Upload Video ---
            if media_type == "Video File":
                uploaded_video = st.file_uploader("Upload Lecture Video File", type=['mp4', 'avi', 'mov'])

                if uploaded_video is not None:
                    os.makedirs("data", exist_ok=True)
                    temp_video_path = os.path.join("data", "temp_video.mp4")
                    processed_video_path = os.path.join("data", "processed_video.mp4")
                    
                    with open(temp_video_path, "wb") as f:
                        f.write(uploaded_video.read())

                    # إنشاء عامودين لعرض الفيديو الأصلي والفيديو المعالج
                    col_input_vid, col_output_vid = st.columns(2, gap="medium")

                    with col_input_vid:
                        st.write("##### 📹 Original Input Video")
                        st.video(temp_video_path)

                    if st.button("🚀 Start Video Analysis", type="primary", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(progress):
                            progress_bar.progress(progress)
                            status_text.text(f"⚡ Processing Video Frames & Drawing Bounding Boxes... {int(progress * 100)}%")

                        processor = VideoProcessor()
                        processed_file_result = processor.process_video_file(
                            video_path=temp_video_path,
                            class_id=selected_class_id,
                            output_path=processed_video_path,
                            progress_callback=update_progress
                        )

                        if processed_file_result and os.path.exists(processed_file_result):
                            # حفظ المسار في session_state ليظل مظهراً في الواجهة
                            st.session_state["processed_video_path"] = processed_file_result
                            st.session_state["processing_success"] = True

                    # عرض الفيديو المعالج بصفة دائمة بمجرد انتهاء المعالجة
                    if st.session_state.get("processing_success", False):
                        st.balloons()
                        st.success("✅ Video processing completed! Attendance & Behavioral events recorded.")
                        
                        with col_output_vid:
                            st.write("##### 🎯 AI Detection & Analytics Video")
                            st.video(st.session_state["processed_video_path"])

            # --- Option B: Upload Image ---
            elif media_type == "Classroom Image":
                uploaded_img = st.file_uploader("Upload Classroom Image Snapshot", type=['jpg', 'jpeg', 'png'])

                if uploaded_img is not None:
                    os.makedirs("data", exist_ok=True)
                    temp_img_path = os.path.join("data", "temp_class.jpg")
                    
                    with open(temp_img_path, "wb") as f:
                        f.write(uploaded_img.read())

                    col_orig, col_res = st.columns(2, gap="medium")
                    
                    with col_orig:
                        st.write("##### Original Snapshot")
                        st.image(temp_img_path, use_container_width=True)

                    if st.button("🔍 Analyze Snapshot & Detect Behaviors", type="primary", use_container_width=True):
                        with st.spinner("Analyzing faces and student engagement behaviors..."):
                            processor = VideoProcessor()
                            success, processed_img = processor.process_image_file(
                                image_path=temp_img_path,
                                class_id=selected_class_id
                            )

                        if success and processed_img is not None:
                            st.success("✅ Image processed! Attendance & Events recorded in database.")
                            with col_res:
                                st.write("##### AI Detection & Recognition Overlay")
                                st.image(processed_img, caption="Processed Image Result", use_container_width=True)
                        else:
                            st.error("❌ Failed to process the image. Please try another snapshot.")

    # --- Page 3: Dashboard ---
    elif selected == "Analytics Dashboard":
        render_dashboard_page()

    # --- Page 4: Subjects & Classes Management ---
    elif selected == "Manage Subjects & Classes":
        st.title("📚 Academic Structure Management")
        st.caption("Set up new academic subjects and organize lecture sessions.")
        st.markdown("---")

        col_add_sub, col_add_class = st.columns(2, gap="large")
        
        with col_add_sub:
            st.subheader("➕ Add New Subject")
            sub_name = st.text_input("Subject Name:", placeholder="e.g., Computer Vision")
            
            if st.button("Save Subject", use_container_width=True):
                if sub_name.strip():
                    sub_id = add_subject(sub_name.strip())
                    st.success(f"✅ Subject '{sub_name}' added successfully! (ID: {sub_id})")
                    st.rerun()
                else:
                    st.error("❌ Please enter a valid subject name.")

        with col_add_class:
            st.subheader("🎓 Create Class Session")
            conn = get_connection()
            subs_df = pd.read_sql_query("SELECT * FROM Subjects", conn)
            conn.close()

            if not subs_df.empty:
                sub_options = {row['Name']: row['Sub_ID'] for _, row in subs_df.iterrows()}
                selected_sub = st.selectbox("Select Subject:", list(sub_options.keys()), key="sub_select_box")
                
                if st.button("Create New Class Session", use_container_width=True):
                    class_id = add_class(sub_options[selected_sub])
                    st.success(f"✅ New Class session created for '{selected_sub}'! (Class ID: {class_id})")
                    st.rerun()
            else:
                st.info("ℹ️ Please add at least one subject first to enable class creation.")

        st.markdown("---")
        
        # Display Current Structure Table
        st.subheader("📋 Registered Subjects & Class Sessions")
        conn = get_connection()
        overview_df = pd.read_sql_query("""
            SELECT c.Class_ID, s.Sub_ID, s.Name as Subject_Name 
            FROM Classes c 
            JOIN Subjects s ON c.Sub_ID = s.Sub_ID
            ORDER BY c.Class_ID DESC
        """, conn)
        conn.close()

        if not overview_df.empty:
            st.dataframe(overview_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No class sessions recorded yet.")

if __name__ == "__main__":
    run_app()