from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.ops import scale_image
from collections import Counter

class VehicleSegmentor:
    def __init__(self, model_path: str, labels: dict, colors: list) -> None:
        self.model = YOLO(model_path)
        self.labels = labels
        self.colors = colors

    def classify_image(self, image, conf):
        result = self.model(image, conf=conf)[0]

        # If nothing is detected, return Nones
        if result.masks is None:
            return None, None, None, None, None
        
        # Detection
        cls = result.boxes.cls.cpu().numpy()
        probs = result.boxes.conf.cpu().numpy() 
        boxes = result.boxes.xyxy.cpu().numpy() 
        labels = result.names

        # Segmentation
        masks = result.masks.data.cpu().numpy()
        masks = np.moveaxis(masks, 0, -1)

        # Rescale masks to original image
        masks = scale_image(masks, result.masks.orig_shape)
        masks = np.moveaxis(masks, -1, 0)

        return boxes, masks, cls, probs, labels

    def overlay_mask(self, image, mask, color, alpha):
        """
        Crates a prediction mask overlay that will be added to the image later
        """
        color = color[::-1]
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
        return image_combined

    def create_segmented_image(self, image, cls, masks):
        """
        Overlays the prediction mask over the image
        """
        image_with_masks = np.copy(image)
        for i, mask in enumerate(masks):
            image_with_masks = self.overlay_mask(image_with_masks, mask, color=self.colors[int(cls[i])], alpha=0.5)
        return image_with_masks

    def predict(self, image):
        """
        Wrapper for the segmentation logic
        """
        boxes, masks, cls, probs, labels = self.classify_image(image, conf=0.60)
        if masks is None:
            return image, {}
        counter = Counter(list(cls.astype("int")))
        counted = list(counter.items())
        counted = dict(map(lambda x: (labels[x[0]], x[1]), counted))
        segmented_image = self.create_segmented_image(image, cls, masks)
        return segmented_image, counted