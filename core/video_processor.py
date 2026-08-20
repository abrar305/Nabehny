import cv2
import numpy as np
import datetime
from datetime import timedelta
import os
import subprocess
from ultralytics import YOLO

from core.face_engine import FaceEngine
from database import (
    get_all_known_faces,
    record_attendance,
    record_event,
    get_connection,
)


class VideoProcessor:
    """
    Computer Vision pipeline:

        Video / Image
          ↓
        YOLOv8n Person Detection (Full Body) + ByteTrack (for video)
          ↓
        Persistent Person Track IDs / Detected Persons
          ↓
        Face Recognition
          ↓
        Track ID / Person → Student ID
          ↓
        Best.pt YOLO Event Detection
          ↓
        Object → Person Track Association
          ↓
        Temporal Event Engine
          ↓
        Attendance + Events Database
          ↓
        Processed Output
    """

    def __init__(
        self,
        person_model_path="yolov8n.pt",
        event_model_path="best.pt",
        sim_threshold=0.5,
        face_recognition_interval=5,
        track_conf=0.25,
        event_buffer_frames=15,
    ):
        # موديل مخصص لاكتشاف الأشخاص وتتبعهم بالكامل (Full Body)[cite: 2]
        self.person_model = YOLO(person_model_path)
        
        # موديل مخصص لاكتشاف الـ Events (مثل رافع ايده، بياكل، إلخ)[cite: 2]
        self.event_model = YOLO(event_model_path)

        self.face_engine = FaceEngine(sim_threshold=sim_threshold)
        self.reload_known_faces()

        # -----------------------------
        # Tracking / identity state
        # -----------------------------
        self.track_identities = {}
        self.track_last_seen = {}
        self.face_recognition_interval = max(1, int(face_recognition_interval))
        self.track_conf = float(track_conf)

        # -----------------------------
        # Temporal event state
        # -----------------------------
        self.active_events = {}
        self.event_buffer_frames = int(event_buffer_frames)

    # =========================================================
    # FACE DATABASE
    # =========================================================

    def reload_known_faces(self):
        self.known_students = get_all_known_faces()
        self.known_ids, self.known_matrix = (
            self.face_engine.prepare_known_faces(self.known_students)
        )

    # =========================================================
    # TIME
    # =========================================================

    def _frame_to_timestamp(self, frame_idx, fps, base_time=None):
        seconds = int(frame_idx / fps) if fps > 0 else 0

        if base_time:
            return (
                base_time + timedelta(seconds=seconds)
            ).strftime("%Y-%m-%d %H:%M:%S")

        return str(timedelta(seconds=seconds))

    # =========================================================
    # GEOMETRY HELPERS
    # =========================================================

    @staticmethod
    def _bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = map(float, box_a)
        bx1, by1, bx2, by2 = map(float, box_b)

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union

    @staticmethod
    def _center_inside_bbox(point, bbox):
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    def _object_belongs_to_person(self, object_bbox, person_bbox):
        ox1, oy1, ox2, oy2 = object_bbox
        px1, py1, px2, py2 = person_bbox

        object_center = (
            (ox1 + ox2) / 2.0,
            (oy1 + oy2) / 2.0,
        )

        person_w = max(1.0, px2 - px1)
        person_h = max(1.0, py2 - py1)

        # 1. منع التداخل الأفقي: يجب أن يكون مركز الكائن/الحدث واقعاً ضمن النطاق الأفقي للشخص حصراً
        # مع سماح بسيط جداً (5%) لمنع الأخطاء الطفيفة في الحدود
        if not (px1 - person_w * 0.05 <= object_center[0] <= px2 + person_w * 0.05):
            return False, float("-inf")

        # 2. السماح بالتوسع الرأسي فقط (للأسفل للأعلى) إذا كان الحدث أعلى الرأس أو أسفل الجذع بقليل
        expanded_person_bbox = [
            px1,  # إلغاء التوسع الأفقي لليسار لمنع التداخل مع الطالب المجاور
            py1 - person_h * 0.10,  # سماح بسيط للأعلى (مثل اليد المرفوعة)
            px2,  # إلغاء التوسع الأفقي لليمين
            py2 + person_h * 0.15,  # سماح للأسفل
        ]

        center_inside = self._center_inside_bbox(
            object_center,
            expanded_person_bbox,
        )

        iou = self._bbox_iou(object_bbox, person_bbox)

        person_center = (
            (px1 + px2) / 2.0,
            (py1 + py2) / 2.0,
        )

        distance = np.sqrt(
            (object_center[0] - person_center[0]) ** 2
            + (object_center[1] - person_center[1]) ** 2
        )

        normalized_distance = distance / max(person_w, person_h)

        if center_inside:
            return True, 1.0 - normalized_distance

        if iou >= 0.05:
            return True, iou

        return False, float("-inf")

    def _match_object_to_person(
        self,
        object_bbox,
        tracked_persons,
    ):
        best_person = None
        best_score = float("-inf")

        for track_id, person in tracked_persons.items():
            matched, score = self._object_belongs_to_person(
                object_bbox,
                person["bbox"],
            )

            if matched and score > best_score:
                best_score = score
                best_person = person

        return best_person

    # =========================================================
    # FACE → TRACK ASSOCIATION
    # =========================================================

    def _match_face_to_person_track(self, face_bbox, tracked_persons):
        best_person = None
        best_score = float("-inf")

        fx1, fy1, fx2, fy2 = face_bbox
        face_center = (
            (fx1 + fx2) / 2.0,
            (fy1 + fy2) / 2.0,
        )

        for track_id, person in tracked_persons.items():
            person_bbox = person["bbox"]
            px1, py1, px2, py2 = person_bbox

            person_center = (
                (px1 + px2) / 2.0,
                (py1 + py2) / 2.0,
            )

            iou = self._bbox_iou(face_bbox, person_bbox)
            center_inside = self._center_inside_bbox(
                face_center,
                person_bbox,
            )

            distance = np.sqrt(
                (face_center[0] - person_center[0]) ** 2
                + (face_center[1] - person_center[1]) ** 2
            )

            person_h = max(1.0, py2 - py1)
            normalized_distance = distance / person_h

            score = (
                (2.0 if center_inside else 0.0)
                + iou * 5.0
                - normalized_distance * 0.25
            )

            if score > best_score:
                best_score = score
                best_person = person

        return best_person

    # =========================================================
    # DRAWING
    # =========================================================

    def _draw_person_track(self, frame, person):
        x1, y1, x2, y2 = map(int, person["bbox"])

        track_id = person.get("track_id", 0)
        stu_id = person.get("stu_id")
        name = person.get("name")
        face_bbox = person.get("face_bbox")

        # 1. رسم مربع الشخص بالكامل (Full Body Box)[cite: 2]
        if stu_id is not None and name:
            label = f"{name} | Track {track_id}" if "track_id" in person else f"{name}"
            color = (0, 255, 0)
        else:
            label = f"Person | Track {track_id}" if "track_id" in person else "Person"
            color = (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 2. رسم مربع الوجه إذا كان موجوداً بداخله[cite: 2]
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = map(int, face_bbox)
            face_color = (0, 255, 0) if stu_id is not None else (0, 0, 255)
            
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)
            
            face_label = f"{name}" if stu_id is not None else "Unknown Face"
            cv2.putText(frame, face_label, (fx1, max(fy1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, face_color, 2)

    # =========================================================
    # EVENT STATE MANAGEMENT
    # =========================================================

    def _update_event(
        self,
        track_id,
        event_type,
        current_timestamp,
        frame_idx,
        confidence,
        bbox,
        student_name,
    ):
        event_key = (track_id, event_type)

        if event_key not in self.active_events:
            self.active_events[event_key] = {
                "track_id": track_id,
                "start_time": current_timestamp,
                "confidence": float(confidence),
                "last_seen_frame": frame_idx,
                "bbox": tuple(map(int, bbox)),
                "student_name": student_name,
            }
        else:
            data = self.active_events[event_key]

            data["last_seen_frame"] = frame_idx
            data["confidence"] = max(
                data["confidence"],
                float(confidence),
            )
            data["bbox"] = tuple(map(int, bbox))
            data["student_name"] = student_name

    def _close_expired_events(
        self,
        frame_idx,
        fps,
        class_id,
        base_time,
    ):
        ended_events = []

        for event_key, data in list(self.active_events.items()):
            if (
                frame_idx - data["last_seen_frame"]
                > self.event_buffer_frames
            ):
                track_id = data["track_id"]
                person = self.track_identities.get(track_id)

                if person and person.get("stu_id") is not None:
                    stu_id = int(person["stu_id"])

                    end_timestamp = self._frame_to_timestamp(
                        data["last_seen_frame"],
                        fps,
                        base_time=base_time,
                    )

                    record_event(
                        stu_id,
                        int(class_id),
                        data["event_type"],
                        data["start_time"],
                        end_timestamp,
                        data["confidence"],
                    )

                ended_events.append(event_key)

        for event_key in ended_events:
            self.active_events.pop(event_key, None)

    # =========================================================
    # VIDEO PROCESSING
    # =========================================================

    def process_video_file(
        self,
        video_path,
        class_id,
        output_path="output_processed.mp4",
        progress_callback=None,
    ):
        self.reload_known_faces()

        self.track_identities.clear()
        self.track_last_seen.clear()
        self.active_events.clear()

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_output = os.path.join("data", "temp_processed_video.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            temp_output,
            fourcc,
            fps,
            (width, height),
        )

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT Stu_ID, Name FROM Students")
        student_names = {
            row[0]: row[1]
            for row in cursor.fetchall()
        }
        conn.close()

        frame_idx = 0
        start_datetime = datetime.datetime.now()
        attended_students = set()

        try:
            cv2.namedWindow("Classroom AI - Processing", cv2.WINDOW_NORMAL)
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                frame_idx += 1

                current_timestamp = self._frame_to_timestamp(
                    frame_idx,
                    fps,
                    base_time=start_datetime,
                )

                display_frame = frame.copy()

                # =================================================
                # 1. YOLOv8n FULL PERSON DETECTION + TRACKING[cite: 2]
                # =================================================

                results = self.person_model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self.track_conf,
                    verbose=False,
                )[0]

                tracked_persons = {}

                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    cls_ids = results.boxes.cls.int().cpu().numpy()
                    confidences = results.boxes.conf.cpu().numpy()

                    if results.boxes.id is not None:
                        track_ids = (
                            results.boxes.id.int().cpu().numpy()
                        )
                    else:
                        track_ids = np.full(
                            len(boxes),
                            -1,
                            dtype=int,
                        )

                    for bbox, cls_id, conf, track_id in zip(
                        boxes,
                        cls_ids,
                        confidences,
                        track_ids,
                    ):
                        class_name = str(
                            self.person_model.names[cls_id]
                        ).lower()

                        if class_name != "person":
                            continue

                        if track_id < 0:
                            continue

                        track_id = int(track_id)

                        tracked_persons[track_id] = {
                            "track_id": track_id,
                            "bbox": tuple(map(int, bbox)),
                            "confidence": float(conf),
                            "stu_id": None,
                            "name": None,
                            "face_bbox": None,
                        }

                        if track_id in self.track_identities:
                            cached = self.track_identities[track_id]

                            tracked_persons[track_id][
                                "stu_id"
                            ] = cached.get("stu_id")

                            tracked_persons[track_id][
                                "name"
                            ] = cached.get("name")
                            
                            tracked_persons[track_id][
                                "face_bbox"
                            ] = cached.get("face_bbox")

                        self.track_last_seen[track_id] = frame_idx

                # =================================================
                # 2. PERIODIC FACE RECOGNITION[cite: 2]
                # =================================================

                should_recognize = (
                    frame_idx % self.face_recognition_interval == 0
                    or frame_idx == 1
                )

                if should_recognize and tracked_persons:

                    face_results = self.face_engine.process_frame(
                        frame,
                        self.known_ids,
                        self.known_matrix,
                    )

                    for face in face_results:
                        stu_id = face.get("stu_id")
                        sim = float(face.get("confidence", 0.0))
                        face_bbox = face.get("bbox")

                        if face_bbox is None:
                            continue

                        matched_track = self._match_face_to_person_track(
                            face_bbox,
                            tracked_persons,
                        )

                        if matched_track is not None:
                            track_id = matched_track["track_id"]
                            
                            if stu_id is not None:
                                stu_id = int(stu_id)
                                stu_name = student_names.get(stu_id, f"ID: {stu_id}")
                                
                                self.track_identities[track_id] = {
                                    "track_id": track_id,
                                    "stu_id": stu_id,
                                    "name": stu_name,
                                    "face_confidence": sim,
                                    "face_bbox": face_bbox,
                                    "last_identity_update": frame_idx,
                                }

                                tracked_persons[track_id]["stu_id"] = stu_id
                                tracked_persons[track_id]["name"] = stu_name
                                tracked_persons[track_id]["face_bbox"] = face_bbox

                                if stu_id not in attended_students:
                                    record_attendance(
                                        stu_id,
                                        int(class_id),
                                        "Present",
                                        current_timestamp,
                                        float(sim),
                                    )
                                    attended_students.add(stu_id)
                            else:
                                self.track_identities[track_id] = {
                                    "track_id": track_id,
                                    "stu_id": None,
                                    "name": "Unknown",
                                    "face_bbox": face_bbox,
                                    "last_identity_update": frame_idx,
                                }
                                tracked_persons[track_id]["face_bbox"] = face_bbox

                        if stu_id not in attended_students and stu_id is not None:
                            record_attendance(
                                stu_id,
                                int(class_id),
                                "Present",
                                current_timestamp,
                                float(sim),
                            )

                            attended_students.add(stu_id)

                # =================================================
                # 3. DRAW PERSON TRACKS (FULL BODY + FACE)[cite: 2]
                # =================================================

                for person in tracked_persons.values():
                    self._draw_person_track(
                        display_frame,
                        person,
                    )

                # =================================================
                # 4. BEST.PT EVENT DETECTION[cite: 2]
                # =================================================

                event_results = self.event_model(
                    frame,
                    verbose=False,
                )[0]

                frame_unique_events = {}

                if event_results.boxes is not None and len(event_results.boxes) > 0:
                    boxes = event_results.boxes.xyxy.cpu().numpy()
                    cls_ids = event_results.boxes.cls.int().cpu().numpy()
                    confidences = event_results.boxes.conf.cpu().numpy()

                    for bbox, cls_id, conf in zip(
                        boxes,
                        cls_ids,
                        confidences,
                    ):
                        event_type = str(
                            self.event_model.names[cls_id]
                        )

                        if event_type.lower() in [
                            "person",
                            "chair",
                            "table",
                            "desk",
                        ]:
                            continue

                        matched_person = (
                            self._match_object_to_person(
                                bbox,
                                tracked_persons,
                            )
                        )

                        ox1, oy1, ox2, oy2 = map(
                            int,
                            bbox,
                        )

                        if matched_person is None:
                            cv2.rectangle(
                                display_frame,
                                (ox1, oy1),
                                (ox2, oy2),
                                (0, 255, 255),
                                2,
                            )

                            cv2.putText(
                                display_frame,
                                f"Activity: {event_type}",
                                (
                                    ox1,
                                    max(oy1 - 5, 15),
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 255, 255),
                                2,
                            )

                            continue

                        track_id = matched_person[
                            "track_id"
                        ]
                        stu_id = matched_person.get("stu_id")

                        if stu_id is None:
                            label = (
                                f"Track {track_id}: "
                                f"{event_type}"
                            )

                            cv2.rectangle(
                                display_frame,
                                (ox1, oy1),
                                (ox2, oy2),
                                (0, 165, 255),
                                2,
                            )

                            cv2.putText(
                                display_frame,
                                label,
                                (
                                    ox1,
                                    max(oy1 - 5, 15),
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 165, 255),
                                2,
                            )

                            continue

                        event_key = (
                            track_id,
                            event_type,
                        )

                        if (
                            event_key not in frame_unique_events
                            or conf
                            > frame_unique_events[event_key][
                                "conf"
                            ]
                        ):
                            frame_unique_events[
                                event_key
                            ] = {
                                "conf": float(conf),
                                "bbox": (
                                    ox1,
                                    oy1,
                                    ox2,
                                    oy2,
                                ),
                                "track_id": track_id,
                                "stu_id": int(stu_id),
                                "student_name": matched_person[
                                    "name"
                                ],
                                "event_type": event_type,
                            }

                # =================================================
                # 5. UPDATE TEMPORAL EVENTS[cite: 2]
                # =================================================

                for (
                    event_key,
                    ev_info,
                ) in frame_unique_events.items():

                    track_id, event_type = event_key

                    self._update_event(
                        track_id=track_id,
                        event_type=event_type,
                        current_timestamp=current_timestamp,
                        frame_idx=frame_idx,
                        confidence=ev_info["conf"],
                        bbox=ev_info["bbox"],
                        student_name=ev_info[
                            "student_name"
                        ],
                    )

                    self.active_events[event_key][
                        "event_type"
                    ] = event_type

                    ox1, oy1, ox2, oy2 = ev_info[
                        "bbox"
                    ]

                    obj_label = (
                        f"{ev_info['student_name']}: "
                        f"{event_type}"
                    )

                    cv2.rectangle(
                        display_frame,
                        (ox1, oy1),
                        (ox2, oy2),
                        (255, 165, 0),
                        2,
                    )

                    cv2.putText(
                        display_frame,
                        obj_label,
                        (
                            ox1,
                            max(oy1 - 5, 15),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 165, 0),
                        2,
                    )

                # =================================================
                # 6. CLOSE EXPIRED EVENTS[cite: 2]
                # =================================================

                self._close_expired_events(
                    frame_idx=frame_idx,
                    fps=fps,
                    class_id=class_id,
                    base_time=start_datetime,
                )

                # =================================================
                # [تعديلك هنا] أضف أي تعديلات إضافية على الإطار هنا
                # =================================================

                # =================================================
                # 7. WRITE FRAME[cite: 2]
                # =================================================
                cv2.imshow("Classroom AI - Processing", display_frame)
                
                if cv2.getWindowProperty("Classroom AI - Processing", cv2.WND_PROP_VISIBLE) >= 1:
                     pass 

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Processing stopped by user.")
                    break
                
                out.write(display_frame)

                if (
                    progress_callback
                    and total_frames > 0
                ):
                    progress_callback(
                        frame_idx / total_frames
                    )

            final_timestamp = self._frame_to_timestamp(
                frame_idx,
                fps,
                base_time=start_datetime,
            )

            for (
                event_key,
                data,
            ) in list(self.active_events.items()):

                track_id = data["track_id"]
                person = self.track_identities.get(
                    track_id
                )

                if (
                    person
                    and person.get("stu_id") is not None
                ):
                    record_event(
                        int(person["stu_id"]),
                        int(class_id),
                        data["event_type"],
                        data["start_time"],
                        final_timestamp,
                        data["confidence"],
                    )

            self.active_events.clear()

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        # =========================================================
        # 9. CONVERT VIDEO TO H.264[cite: 2]
        # =========================================================

        abs_temp_output = os.path.abspath(
            temp_output
        )
        abs_output_path = os.path.abspath(
            output_path
        )

        try:
            import imageio_ffmpeg as ffmpeg

            ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                abs_temp_output,
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                abs_output_path,
            ]

            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            if os.path.exists(abs_temp_output):
                os.remove(abs_temp_output)

            return abs_output_path

        except Exception as e:
            print(
                f"[WARNING] Conversion failed: {e}. "
                "Falling back to original output."
            )

            if os.path.exists(abs_temp_output):
                return abs_temp_output

            return abs_output_path

    # =========================================================
    # IMAGE PROCESSING[cite: 2]
    # =========================================================

    def process_image_file(self, image_path, class_id):
        self.reload_known_faces()

        frame = cv2.imread(image_path)

        if frame is None:
            return False, None

        annotated_frame = frame.copy()

        now_str = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT Stu_ID, Name FROM Students"
        )

        student_names = {
            row[0]: row[1]
            for row in cursor.fetchall()
        }

        conn.close()

        # =====================================================
        # 1. YOLOv8n FULL PERSON DETECTION ON IMAGE[cite: 2]
        # =====================================================
        person_results = self.person_model(
            frame,
            verbose=False,
        )[0]

        detected_persons = {}

        if person_results.boxes is not None and len(person_results.boxes) > 0:
            boxes = person_results.boxes.xyxy.cpu().numpy()
            cls_ids = person_results.boxes.cls.int().cpu().numpy()
            confidences = person_results.boxes.conf.cpu().numpy()

            for idx, (bbox, cls_id, conf) in enumerate(zip(boxes, cls_ids, confidences)):
                class_name = str(self.person_model.names[cls_id]).lower()
                if class_name != "person":
                    continue

                detected_persons[idx] = {
                    "track_id": idx,
                    "bbox": tuple(map(int, bbox)),
                    "confidence": float(conf),
                    "stu_id": None,
                    "name": None,
                    "face_bbox": None,
                }

        # =====================================================
        # 2. FACE RECOGNITION & ASSOCIATION TO PERSONS[cite: 2]
        # =====================================================

        face_results = self.face_engine.process_frame(
            frame,
            self.known_ids,
            self.known_matrix,
        )

        for face in face_results:
            stu_id = face.get("stu_id")
            sim = float(
                face.get("confidence", 0.0)
            )
            face_bbox = face.get("bbox")

            if face_bbox is None:
                continue

            matched_person = self._match_face_to_person_track(
                face_bbox,
                detected_persons,
            )

            if matched_person is not None:
                if stu_id is not None:
                    stu_id = int(stu_id)
                    name = student_names.get(
                        stu_id,
                        f"ID: {stu_id}",
                    )

                    matched_person["stu_id"] = stu_id
                    matched_person["name"] = name
                    matched_person["face_bbox"] = face_bbox

                    try:
                        record_attendance(
                            stu_id=stu_id,
                            class_id=int(class_id),
                            status="Present",
                            timestamp=now_str,
                            confidence=round(sim, 2),
                        )
                    except Exception as e:
                        print(
                            f"[ERROR] Attendance insert failed: {e}"
                        )
                else:
                    matched_person["name"] = "Unknown"
                    matched_person["face_bbox"] = face_bbox

        # Draw all detected full-body persons (with matched faces inside)[cite: 2]
        for person in detected_persons.values():
            self._draw_person_track(annotated_frame, person)

        # =====================================================
        # 3. BEST.PT EVENT DETECTION (IMAGE)[cite: 2]
        # =====================================================

        yolo_results = self.event_model(
            frame,
            verbose=False,
        )[0]

        if (
            yolo_results.boxes is not None
            and len(yolo_results.boxes) > 0
        ):

            boxes = (
                yolo_results.boxes.xyxy
                .cpu()
                .numpy()
            )

            cls_ids = (
                yolo_results.boxes.cls
                .int()
                .cpu()
                .numpy()
            )

            confidences = (
                yolo_results.boxes.conf
                .cpu()
                .numpy()
            )

            unique_detected_events = {}

            for bbox, cls_id, conf in zip(
                boxes,
                cls_ids,
                confidences,
            ):
                event_type = str(
                    self.event_model.names[cls_id]
                )

                if event_type.lower() in [
                    "person",
                    "chair",
                    "table",
                    "desk",
                ]:
                    continue

                ox1, oy1, ox2, oy2 = map(
                    int,
                    bbox,
                )

                matched_student = (
                    self._match_object_to_person(
                        bbox,
                        detected_persons,
                    )
                )

                if matched_student and matched_student.get("stu_id") is not None:
                    stu_id = int(
                        matched_student["stu_id"]
                    )

                    event_key = (
                        stu_id,
                        event_type,
                    )

                    if (
                        event_key
                        not in unique_detected_events
                        or conf
                        > unique_detected_events[
                            event_key
                        ]["conf"]
                    ):
                        unique_detected_events[
                            event_key
                        ] = {
                            "conf": float(conf),
                            "bbox": (
                                ox1,
                                oy1,
                                ox2,
                                oy2,
                            ),
                            "student": matched_student,
                            "event_type": event_type,
                        }

                else:
                    obj_color = (255, 165, 0)

                    cv2.rectangle(
                        annotated_frame,
                        (ox1, oy1),
                        (ox2, oy2),
                        obj_color,
                        2,
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Activity: {event_type}",
                        (
                            ox1,
                            max(oy1 - 5, 15),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        obj_color,
                        2,
                    )

            # =================================================
            # 4. RECORD IMAGE EVENTS[cite: 2]
            # =================================================

            for (
                (stu_id, event_type),
                ev_data,
            ) in unique_detected_events.items():

                try:
                    record_event(
                        stu_id=stu_id,
                        class_id=int(class_id),
                        event_type=event_type,
                        start_time=now_str,
                        end_time=now_str,
                        confidence=ev_data["conf"],
                    )

                except Exception as e:
                    print(
                        f"[ERROR] Event insert failed: {e}"
                    )

                ox1, oy1, ox2, oy2 = (
                    ev_data["bbox"]
                )

                matched_name = ev_data[
                    "student"
                ]["name"]

                obj_color = (255, 165, 0)

                obj_label = (
                    f"{matched_name}: "
                    f"{event_type}"
                )

                cv2.rectangle(
                    annotated_frame,
                    (ox1, oy1),
                    (ox2, oy2),
                    obj_color,
                    2,
                )

                cv2.putText(
                    annotated_frame,
                    obj_label,
                    (
                        ox1,
                        max(oy1 - 5, 15),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    obj_color,
                    2,
                )

        annotated_frame_rgb = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB,
        )

        return True, annotated_frame_rgb