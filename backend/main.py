from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import tempfile
from ultralytics import YOLO
import cv2
import shutil

app = FastAPI()

# Enable CORS for all origins (allows frontend on different ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static folder exists and mount it for images
os.makedirs("static/animals", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLO model - make sure yolov8n.pt is in backend folder
model = YOLO("yolov8n.pt")

class ClassificationResponse(BaseModel):
    name: str
    images: List[str]

class FinalResponse(BaseModel):
    animals: List[ClassificationResponse]
    location: dict

@app.post("/classify")
async def classify(video: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_path = temp_file.name
        temp_file.write(await video.read())

    # Clear old images
    shutil.rmtree("static/animals")
    os.makedirs("static/animals", exist_ok=True)

    try:
        cap = cv2.VideoCapture(temp_path)
        seen_classes = {}
        frame_count = 0
        image_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:  # process every 30th frame
                results = model.predict(frame, imgsz=640, conf=0.5)
                for result in results:
                    if hasattr(result, 'names'):
                        for box in result.boxes:
                            cls = int(box.cls)
                            label = result.names[cls]
                            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                            cropped = frame[ymin:ymax, xmin:xmax]
                            filename = f"static/animals/{label}_{image_counter}.jpg"
                            cv2.imwrite(filename, cropped)

                            if label not in seen_classes:
                                seen_classes[label] = []
                            seen_classes[label].append(f"/static/animals/{label}_{image_counter}.jpg")
                            image_counter += 1
            frame_count += 1

        cap.release()

        location = {
            "latitude": 12.9716,
            "longitude": 77.5946
        }

        animals = [
            {"name": name, "images": paths}
            for name, paths in seen_classes.items()
        ]

    except Exception as e:
        print("Error processing video:", e)
        return {"animals": [], "location": {}}
    finally:
        os.remove(temp_path)

    return {"animals": animals, "location": location}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
