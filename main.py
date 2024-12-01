import cv2
import numpy as np
import base64
import json
from markupsafe import escape

def detect_black_dots(request):
    # Parse the image from the request
    try:
        request_json = request.get_json()
        image_data = request_json.get("image")  # Base64-encoded image
        grid_rows = request_json.get("grid_rows", 5)  # Default grid rows
        grid_cols = request_json.get("grid_cols", 5)  # Default grid cols

        # Decode the image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Image dimensions
        height, width = binary.shape
        cell_height = height // grid_rows
        cell_width = width // grid_cols

        # Detect black dots
        black_dots = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            row = min(center_y // cell_height, grid_rows - 1) + 1
            col = min(center_x // cell_width, grid_cols - 1) + 1
            black_dots.append({"row": row, "col": col})

        # Return the positions of the black dots
        return json.dumps({"black_dots": black_dots})

    except Exception as e:
        return json.dumps({"error": str(e)})
