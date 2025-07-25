import cv2
import numpy as np
import base64


def process_request_image(file: bytes) -> np.ndarray:
    """
    Converts a raw image byte stream from an HTTP request to an OpenCV BGR image.

    Args:
        file (bytes): Raw bytes from an uploaded image file.

    Returns:
        np.ndarray: Decoded OpenCV image in BGR format.

    Raises:
        ValueError: If image decoding fails.
    """
    np_image = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode the uploaded image.")

    return image

def process_response_image(image: np.ndarray) -> str:
    """
    Converts an OpenCV BGR image to a base64-encoded PNG string for JSON response.

    Args:
        image (np.ndarray): OpenCV BGR image.

    Returns:
        str: Base64-encoded image string suitable for embedding in a JSON response.

    Raises:
        ValueError: If image encoding fails.
    """
    success, encoded_image = cv2.imencode('.png', image)
    if not success:
        raise ValueError("Could not encode image to PNG format.")

    encoded_bytes = encoded_image.tobytes()
    encoded_base64 = base64.b64encode(encoded_bytes).decode('utf-8')

    return encoded_base64