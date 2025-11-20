**Vehicle Number Plate Detection & Logging**

Real-time number plate detection system using YOLO, PaddleOCR, OpenCV, and MySQL. The system reads vehicle plates from live video, identifies the owner from a database, and logs each detection with date, time, and tracking ID.

⭐ Features

Real-time vehicle detection using YOLO

OCR-based number plate recognition (PaddleOCR)

MySQL owner verification

Auto-logging of detections (date, time, track_id, plate, owner)

Bounding boxes with color (green for known, red for unknown)

Duplicate logs avoided using tracked IDs

🧠 Tech Stack

Python

YOLO (Ultralytics)

PaddleOCR

OpenCV

MySQL

🔧 How It Works

YOLO detects vehicles in each frame

Cropped plate region is passed to OCR

Extracted number is checked in the MySQL table

Owner name is fetched (or marked unknown)

A log entry is inserted into the database

Frame is displayed with detection + owner label

🗄 Database

Database: numberplates_speed4
Tables:

registered_vehicles(numberplate, owner_name)

vehicle_logs1(id, date, time, track_id, numberplate, owner_name)

📦 Installation
pip install -r requirements.txt


Install PaddleOCR:

pip install paddlepaddle paddleocr


Make sure MySQL is running.
Tables are auto-created.

▶️ Run
python new_vehicle.py
