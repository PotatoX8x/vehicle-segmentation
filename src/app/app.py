
import uvicorn

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile
from starlette.responses import FileResponse, JSONResponse
from src.vehicle_segmentation.VehicleSegmentor import VehicleSegmentor
from fastapi.encoders import jsonable_encoder

from src.app.app_utils import process_request_image, process_response_image

# Prediction config
model_path = "models/yolo_seg.pt"
labels = {0: 'Ambulance', 1: 'Bus', 2: 'Car', 3: 'Motorcycle', 4: 'Truck'}
colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11)]

vehicle_segmentor = VehicleSegmentor(model_path, labels, colors)


app = FastAPI()
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('src/app/index.html')

@app.post("/vehicle_segment")
async def image_detect(file: UploadFile):
    file = await file.read()
    image = process_request_image(file)

    segmented_image, counted = vehicle_segmentor.predict(image)
    encoded_image = process_response_image(segmented_image)

    # Counts for each class predicted
    counted = jsonable_encoder(counted)

    response = {
        "image": encoded_image,
        "predicted": counted
    }

    return JSONResponse(content = response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)