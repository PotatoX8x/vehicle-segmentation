
import uvicorn
from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile
from starlette.responses import Response, FileResponse, JSONResponse

from VehicleSegmentor import VehicleSegmentor
import json
import base64
from fastapi.encoders import jsonable_encoder
import time
vehicle_segmentor = VehicleSegmentor()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

def process_request_image(file):
    np_image = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    return image

def process_response_image(image):
    _, encoded_image = cv2.imencode('.png', image)
    encoded_image_bytes = encoded_image.tobytes()
    encoded_image_base64 = base64.b64encode(encoded_image_bytes)
    decoded_image_base64 = encoded_image_base64.decode('utf-8')
    return decoded_image_base64

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/vehicle_segment")
async def image_detect(file: UploadFile):
    file = await file.read()
    image = process_request_image(file)

    segmented_image, counted = vehicle_segmentor.predict(image)
    encoded_image = process_response_image(segmented_image)
    counted = jsonable_encoder(counted)

    response = {
        "image": encoded_image,
        "predicted": counted
    }

    return JSONResponse(content = response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)