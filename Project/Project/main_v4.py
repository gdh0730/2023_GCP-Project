from flask import Flask, request, jsonify
from google.cloud import vision_v1
import numpy as np
import cv2
import io

app = Flask(__name__)

@app.route('/')
def home():
    return "App is running!"

@app.route('/detect_clothing', methods=['POST'])
def detect_clothing_endpoint():
    image_file = request.files['image']
    image_content = image_file.read()
    object_details = detect_clothing_and_get_web_info(image_content)
    return jsonify(object_details)

def detect_clothing_and_get_web_info(image_content):
    client = vision_v1.ImageAnnotatorClient()
    image = vision_v1.Image(content=image_content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    # Convert bytes to OpenCV image
    image_cv = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = image_cv.shape

    # Define a list of fashion-related labels
    fashion_labels = ['Shirt', 'Dress', 'Shoe', 'Hat', 'Jacket', 'Skirt', 'Pants', 'Tie', 'Scarf', 'outerwear']

    # List to store object details
    object_details = []

    # Extract and save fashion-related objects
    for object_ in objects:
        if object_.name in fashion_labels:
            vertices = [(int(vertex.x * width), int(vertex.y * height)) for vertex in object_.bounding_poly.normalized_vertices]
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(np.array(vertices))
            scaled_rect_x, scaled_rect_y, scaled_rect_w, scaled_rect_h = rect_x / width, rect_y / height, rect_w / width, rect_h / height
            cut_region = image_cv[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]

            # Convert the cut region to bytes to be used for web detection
            is_success, buffer = cv2.imencode(".jpg", cut_region)
            io_buf = io.BytesIO(buffer)
            web_detection = get_web_detection(io_buf.getvalue())

            object_details.append({
                'id': object_.name,
                'label': object_.name,
                'bounding_box': {'x': scaled_rect_x, 'y': scaled_rect_y, 'w': scaled_rect_w, 'h': scaled_rect_h},
                'web_entities': web_detection
            })

    return object_details

def get_web_detection(image_content):
    """Get web detection results using Cloud Vision API."""
    client = vision_v1.ImageAnnotatorClient()
    image = vision_v1.Image(content=image_content)
    response = client.web_detection(image=image)
    
    web_entities = []
    for entity in response.web_detection.web_entities:
        web_entities.append({
            'description': entity.description,
            'score': entity.score
        })
    
    return web_entities

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
