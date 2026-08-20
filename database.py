import sqlite3
import numpy as np
import os

DB_PATH = os.path.join("data", "db.sqlite3")

def get_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # 1. جدول Subjects
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Subjects (
            Sub_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL
        )
    """)

    # 2. جدول Classes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Classes (
            Class_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Sub_ID INTEGER NOT NULL,
            FOREIGN KEY (Sub_ID) REFERENCES Subjects (Sub_ID) ON DELETE CASCADE
        )
    """)

    # 3. جدول Students
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Students (
            Stu_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL
        )
    """)

    # 4. جدول Student_Images
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Student_Images (
            Image_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Stu_ID INTEGER NOT NULL,
            Image_Path TEXT NOT NULL,
            Face_Embedding BLOB NOT NULL,
            FOREIGN KEY (Stu_ID) REFERENCES Students (Stu_ID) ON DELETE CASCADE
        )
    """)

    # 5. جدول Attendance مع إضافة UNIQUE لمنع التكرار
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Attendance (
            Attend_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Stu_ID INTEGER NOT NULL,
            Class_ID INTEGER NOT NULL,
            Status TEXT NOT NULL,
            Timestamp TEXT NOT NULL,
            Confidence REAL NOT NULL,
            FOREIGN KEY (Stu_ID) REFERENCES Students (Stu_ID) ON DELETE CASCADE,
            FOREIGN KEY (Class_ID) REFERENCES Classes (Class_ID) ON DELETE CASCADE,
            UNIQUE(Stu_ID, Class_ID)
        )
    """)

    # 6. جدول Student_Events مع تصحيح أسماء الأعمدة (Start_Time / End_Time)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Student_Events (
            Event_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Stu_ID INTEGER NOT NULL,
            Class_ID INTEGER NOT NULL,
            Event_Type TEXT NOT NULL,
            Start_Time TEXT NOT NULL,
            End_Time TEXT NOT NULL,
            Confidence REAL NOT NULL,
            FOREIGN KEY (Stu_ID) REFERENCES Students (Stu_ID) ON DELETE CASCADE,
            FOREIGN KEY (Class_ID) REFERENCES Classes (Class_ID) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()

# Helper Functions

def add_student(name):
    """إضافة طالب جديد وإرجاع الـ Stu_ID"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO Students (Name) VALUES (?)", (name,))
        conn.commit()
        stu_id = cursor.lastrowid
        print(f"[SUCCESS] Student added with ID: {stu_id}")
        return stu_id
    except Exception as e:
        print(f"[ERROR] Failed to add student: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def add_student_image(stu_id, image_path, embedding_array):
    """حفظ صورة الطالب والـ Embedding الخاص بالوجه"""
    if stu_id is None or embedding_array is None:
        print("[ERROR] Cannot save image: Invalid Stu_ID or Embedding!")
        return False

    embedding_bytes = embedding_array.astype(np.float32).tobytes()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO Student_Images (Stu_ID, Image_Path, Face_Embedding) VALUES (?, ?, ?)",
            (stu_id, image_path, embedding_bytes)
        )
        conn.commit()
        print(f"[SUCCESS] Image saved for Stu_ID {stu_id}: {image_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save image path: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_all_known_faces():
    """جلب كل الـ Embeddings لجميع الطلاب دون حساب المتوسط"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Stu_ID, Face_Embedding FROM Student_Images")
    rows = cursor.fetchall()
    conn.close()

    known_students = []
    for stu_id, emb_bytes in rows:
        emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
        # L2 Normalization لكل Vector منفرد
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm

        known_students.append({
            "stu_id": stu_id,
            "embedding": emb_array
        })
        
    return known_students

def add_subject(name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Subjects (Name) VALUES (?)", (name,))
    sub_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return sub_id

def add_class(sub_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Classes (Sub_ID) VALUES (?)", (sub_id,))
    class_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return class_id

def record_attendance(stu_id, class_id, status, timestamp, confidence):
    """تسجيل الحضور ومنع التكرار بسلاسة (تحديث البيانات لو وُجدت سابقاً)"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO Attendance (Stu_ID, Class_ID, Status, Timestamp, Confidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(Stu_ID, Class_ID) DO UPDATE SET
                Status=excluded.Status,
                Timestamp=excluded.Timestamp,
                Confidence=excluded.Confidence
        """, (stu_id, class_id, status, timestamp, confidence))
        conn.commit()
    except Exception as e:
        print(f"[ERROR] Attendance recording failed: {e}")
        conn.rollback()
    finally:
        conn.close()

def record_event(stu_id, class_id, event_type, start_time, end_time, confidence):
    """تسجيل حدث أو نشاط للطالب"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO Student_Events (Stu_ID, Class_ID, Event_Type, Start_Time, End_Time, Confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (stu_id, class_id, event_type, start_time, end_time, confidence))
        conn.commit()
    except Exception as e:
        print(f"[ERROR] Event recording failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")