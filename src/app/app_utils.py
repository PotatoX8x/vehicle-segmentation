import cv2
import numpy as np
import base64


def process_request_image(file):
    """
    Transforms base64 image from request into cv2 object
    """
    np_image = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    return image

def process_response_image(image):
    """
    Transforms a cv2 image object into base64 bytes
    """
    _, encoded_image = cv2.imencode('.png', image)
    encoded_image_bytes = encoded_image.tobytes()
    encoded_image_base64 = base64.b64encode(encoded_image_bytes)
    decoded_image_base64 = encoded_image_base64.decode('utf-8')
    return decoded_image_base64