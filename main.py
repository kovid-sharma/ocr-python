import cv2
import numpy as np
import base64
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/detect_black_dots', methods=['POST'])
def detect_black_dots():
    try:
        # Parse the image from the request
        request_json = request.get_json()
        image_data = request_json.get("image")  # Base64-encoded image
        grid_rows = request_json.get("grid_rows", 5)  # Default grid rows
        grid_cols = request_json.get("grid_cols", 5)  # Default grid cols

        if not image_data:
            return jsonify({"error": "Image data is missing"}), 400

        # Correct padding for Base64 string if necessary
        padding_needed = len(image_data) % 4
        if padding_needed != 0:
            image_data += '=' * (4 - padding_needed)  # Add the necessary padding

        # Decode the image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image data"}), 400

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

        # Detect black dots and organize by row
        dots_map = {}
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            row = min(center_y // cell_height, grid_rows - 1) + 1
            col = min(center_x // cell_width, grid_cols - 1) + 1
            if row not in dots_map:
                dots_map[row] = []
            dots_map[row].append(col)

        # Sort the column indices in each row
        for row in dots_map:
            row= "Row"+row+"-"
            dots_map[row].sort()

        # Return the map of black dots
        return jsonify({"black_dots_map": dots_map})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
