import streamlit as st
import pandas as pd
import plotly.express as px
from database import get_connection

def apply_dashboard_styles():
    st.markdown("""
        <style>
        .metric-card {
            background-color: #1E222D;
            padding: 16px;
            border-radius: 10px;
            border-left: 4px solid #3B82F6;
            margin-bottom: 10px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

def render_dashboard_page():
    apply_dashboard_styles()
    st.title("📊 Class Analytics & Behavioral Dashboard")
    st.caption("Real-time automated insight tracking for class attendance and engagement behaviors.")

    conn = get_connection()

    # --- 1. Class / Lecture Selection ---
    classes_df = pd.read_sql_query("""
        SELECT c.Class_ID, s.Name as Subject_Name 
        FROM Classes c 
        JOIN Subjects s ON c.Sub_ID = s.Sub_ID
    """, conn)

    if classes_df.empty:
        st.warning("⚠️ No lectures or classes recorded in the database yet.")
        conn.close()
        return

    class_options = {f"Class #{row['Class_ID']} — {row['Subject_Name']}": row['Class_ID'] 
                     for _, row in classes_df.iterrows()}
    
    col_select, col_blank = st.columns([2, 1])
    with col_select:
        selected_class_name = st.selectbox("📚 Select Active Lecture / Session:", list(class_options.keys()))
    
    selected_class_id = class_options[selected_class_name]

    # --- Fetch Data ---
    attendance_df = pd.read_sql_query("""
        SELECT s.Stu_ID, s.Name as Student_Name, a.Status, a.Timestamp, a.Confidence
        FROM Attendance a
        JOIN Students s ON a.Stu_ID = s.Stu_ID
        WHERE a.Class_ID = ?
    """, conn, params=(selected_class_id,))

    all_students_df = pd.read_sql_query("SELECT Stu_ID, Name FROM Students", conn)

    # Corrected SQL Column names (Start_Time, End_Time)
    events_df = pd.read_sql_query("""
        SELECT e.Event_ID, s.Name as Student_Name, e.Event_Type, e.Start_Time, e.End_Time, e.Confidence
        FROM Student_Events e
        JOIN Students s ON e.Stu_ID = s.Stu_ID
        WHERE e.Class_ID = ?
    """, conn, params=(selected_class_id,))

    conn.close()

    st.markdown("---")

    # --- Key Performance Metrics (KPIs) ---
    total_students = len(all_students_df) if not all_students_df.empty else 0
    attended_ids = attendance_df['Stu_ID'].tolist() if not attendance_df.empty else []
    present_count = len(attended_ids)
    absent_count = total_students - present_count
    att_rate = round((present_count / total_students * 100), 1) if total_students > 0 else 0.0

    top_behavior = events_df['Event_Type'].mode()[0] if not events_df.empty else "N/A"
    most_active_student = events_df['Student_Name'].mode()[0] if not events_df.empty else "N/A"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("👥 Total Enrolled", total_students)
    m2.metric("✅ Present Students", present_count, f"{att_rate}% Rate")
    m3.metric("🔥 Top Behavior", top_behavior)
    m4.metric("⭐ Most Active Student", most_active_student)

    st.markdown("---")

    # --- Main Navigation Tabs ---
    tab_att, tab_events, tab_logs, tab_export = st.tabs([
        "📋 Attendance Summary", 
        "🎭 Behavioral Analytics", 
        "📝 Event Logs", 
        "📥 Reports Export"
    ])

    # TAB 1: Attendance Report
    with tab_att:
        col_chart, col_table = st.columns([1, 1], gap="large")
        
        if not all_students_df.empty:
            all_students_df['Status'] = all_students_df['Stu_ID'].apply(
                lambda x: 'Present' if x in attended_ids else 'Absent'
            )

            with col_chart:
                fig_att = px.pie(
                    all_students_df, 
                    names='Status', 
                    title="<b>Attendance Ratio</b>", 
                    color='Status',
                    hole=0.4,
                    color_discrete_map={'Present': '#10B981', 'Absent': '#EF4444'}
                )
                fig_att.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_att, use_container_width=True)

            with col_table:
                st.write("##### Student Status List")
                st.dataframe(
                    all_students_df[['Stu_ID', 'Name', 'Status']], 
                    use_container_width=True,
                    hide_index=True
                )

    # TAB 2: Behavior Analytics
    with tab_events:
        if events_df.empty:
            st.info("ℹ️ No student behaviors or events detected for this lecture session.")
        else:
            col_b1, col_b2 = st.columns([1, 1], gap="large")

            with col_b1:
                event_counts = events_df['Event_Type'].value_counts().reset_index()
                event_counts.columns = ['Event_Type', 'Count']
                
                fig_events = px.bar(
                    event_counts, 
                    x='Event_Type', 
                    y='Count', 
                    title="<b>Behavior Frequencies</b>",
                    color='Event_Type',
                    text='Count',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_events, use_container_width=True)

            with col_b2:
                stu_events = events_df['Student_Name'].value_counts().reset_index()
                stu_events.columns = ['Student_Name', 'Event_Count']

                fig_stu = px.bar(
                    stu_events,
                    x='Event_Count',
                    y='Student_Name',
                    orientation='h',
                    title="<b>Behaviors Detected per Student</b>",
                    color='Event_Count',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_stu, use_container_width=True)

    # TAB 3: Event Logs Table
    with tab_logs:
        if events_df.empty:
            st.info("ℹ️ Log is empty.")
        else:
            st.write("##### Detailed Timeline of Class Events")
            st.dataframe(
                events_df.sort_values(by="Start_Time", ascending=False), 
                use_container_width=True,
                hide_index=True
            )

    # TAB 4: Reports & Exports
    with tab_export:
        st.subheader("📄 Export Session Reports")
        st.write("Download formatted CSV files for official record keeping.")

        c1, c2 = st.columns(2)
        
        with c1:
            if not all_students_df.empty:
                att_csv = all_students_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Attendance Report (CSV)",
                    data=att_csv,
                    file_name=f"class_{selected_class_id}_attendance.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with c2:
            if not events_df.empty:
                events_csv = events_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Behavior Log (CSV)",
                    data=events_csv,
                    file_name=f"class_{selected_class_id}_events.csv",
                    mime="text/csv",
                    use_container_width=True
                )