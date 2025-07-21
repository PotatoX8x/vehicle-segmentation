# Vehicle segmentation API

## Environment setup
- Python 3.11
- `pip install -r "requirements.txt"`
- `.env` with `ROBOFLOW_API_KEY`

## Model training
In the notebook `src/training/train.ipynb` a model training script can be found. It downloads the dataset for object detection from Roboflow, trains a detection model, annotates the dataset with segmentaion masks and trains the segmentation model.

## API
Launch the app:

```python -m src.app.app```

The app contains a UI to upload images and get predictions. 

