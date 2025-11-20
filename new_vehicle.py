import cv2
from time import time
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator
from datetime import datetime
import mysql.connector
from paddleocr import PaddleOCR

class SpeedEstimator(BaseSolution):
    def __init__(self, video_source=0, **kwargs):
        super().__init__(**kwargs)
        self.trkd_ids = []  
        self.trk_pt = {}  
        self.trk_pp = {}  
        self.logged_ids = set()  

        self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en')

        self.db_connection = self.connect_to_db()

        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            exit()
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.initialize_region()

    def initialize_region(self):
        """Dynamically initialize speed region based on frame size."""
        self.region = [(0, int(self.frame_height * 0.3)), (self.frame_width, int(self.frame_height * 0.3))]
        print(f"Region initialized dynamically: {self.region}")

    def connect_to_db(self):
        """Connects to MySQL database and ensures the required tables exist."""
        try:
            connection = mysql.connector.connect(
                host="localhost", user="root", password="pass123"
            )
            cursor = connection.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS numberplates_speed4")
            connection.database = "numberplates_speed4"

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS registered_vehicles (
                    numberplate VARCHAR(20) PRIMARY KEY,
                    owner_name VARCHAR(100)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_logs1 (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE,
                    time TIME,
                    track_id INT,
                    numberplate VARCHAR(20),
                    owner_name VARCHAR(100)
                )
            """)

            return connection
        except mysql.connector.Error as err:
            print(f"Database connection error: {err}")
            raise

    def get_owner_name(self, numberplate):
        """Fetches owner name for a given number plate from the database."""
        try:
            cursor = self.db_connection.cursor()
            query = "SELECT owner_name FROM registered_vehicles WHERE numberplate = %s"
            cursor.execute(query, (numberplate,))
            result = cursor.fetchone()

            return result[0] if result else "Unknown"
        except mysql.connector.Error as err:
            print(f"Database query error: {err}")
            return "Unknown"

    def perform_ocr(self, image_array):
        """Performs OCR on a cropped image."""
        if image_array is None or not isinstance(image_array, np.ndarray):
            return ""
        results = self.ocr.ocr(image_array, rec=True)
        return ' '.join([result[1][0] for result in results[0]] if results[0] else "")

    def save_to_database(self, date, time, track_id, numberplate, owner_name):
        """Saves detected data to MySQL database."""
        try:
            cursor = self.db_connection.cursor()
            query = """
                INSERT INTO vehicle_logs1 (date, time, track_id, numberplate, owner_name)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (date, time, track_id, numberplate, owner_name))
            self.db_connection.commit()
        except mysql.connector.Error as err:
            print(f"Error saving to database: {err}")

    def estimate_speed(self, im0):
        """Processes frame and detects vehicle speeds."""
        self.annotator = Annotator(im0, line_width=self.line_width)
        self.extract_tracks(im0)
        current_time = datetime.now()

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)

            x1, y1, x2, y2 = map(int, box)
            cropped_image = np.array(im0)[y1:y2, x1:x2]
            ocr_text = self.perform_ocr(cropped_image).strip()

            owner_name = "Unknown"
            if ocr_text:
                owner_name = self.get_owner_name(ocr_text)

                if track_id not in self.logged_ids:
                    self.save_to_database(
                        current_time.strftime("%Y-%m-%d"),
                        current_time.strftime("%H:%M:%S"),
                        track_id, ocr_text, owner_name
                    )
                    self.logged_ids.add(track_id)

            color = (0, 255, 0) if owner_name != "Unknown" else (0, 0, 255)

            thickness = 2
            cv2.rectangle(im0, (x1, y1), (x2, y2), color, thickness)

            label = f"{ocr_text} ({owner_name})" if ocr_text else "No Plate Detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

            cv2.rectangle(im0, (label_x, label_y - label_size[1]), (label_x + label_size[0], label_y + 5), color, -1)
            cv2.putText(im0, label, (label_x, label_y), font, font_scale, (255, 255, 255), font_thickness)

        return im0

speed_obj = SpeedEstimator(video_source=0, model="best.pt", line_width=2)
count = 0

while True:
    ret, frame = speed_obj.cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (speed_obj.frame_width, speed_obj.frame_height))
    result = speed_obj.estimate_speed(frame)
    print(f"Frame {count} processed.")

    cv2.imshow("RGB", result)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

speed_obj.cap.release()
cv2.destroyAllWindows()
