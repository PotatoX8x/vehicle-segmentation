import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from starlette.responses import FileResponse, JSONResponse

from src.app.exception_handling import handle_exceptions
from src.app.app_utils import process_request_image, process_response_image
from src.vehicle_segmentation.VehicleSegmentor import VehicleSegmentor


# Prediction config
model_path = "models/yolo_seg.pt"
label_config = {
    0: ('Ambulance', (89, 161, 197)), 
    1: ('Bus', (67, 161, 255)), 
    2: ('Car', (19, 222, 24)), 
    3: ('Motorcycle', (186, 55, 2)), 
    4: ('Truck', (167, 146, 11))
}

vehicle_segmentor = VehicleSegmentor(model_path, label_config)


app = FastAPI()

handle_exceptions(
    app=app
)

# Serve static files (styles, scripts)
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

@app.get("/")
async def read_index() -> FileResponse:
    """
    Serves the web interface from index.html at the root path.
    """
    return FileResponse('src/app/index.html')

@app.post("/vehicle_segment")
async def vehicle_segment(file: UploadFile) -> JSONResponse:
    """
    Accepts an image file, segments vehicles in the image,
    and returns the segmented image (base64 encoded) and class counts.

    Args:
        file (UploadFile): Image file uploaded via form-data.

    Returns:
        JSONResponse: {
            "image": base64 string,
            "predicted": dict with class counts
        }
    """
    file = await file.read()
    image = process_request_image(file)
    if image is not None:
        segmented_image, counted = vehicle_segmentor.predict(image)
        encoded_image = process_response_image(segmented_image)

        # Counts for each class predicted
        counted = jsonable_encoder(counted)

        response = {
            "image": encoded_image,
            "predicted": counted
        }
    else:
        response = {
            "image": None,
            "predicted": None
        }

    return JSONResponse(content=response)


# Entrypoint
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)