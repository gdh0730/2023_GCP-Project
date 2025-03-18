from flask import Flask, request, jsonify
from google.cloud import vision_v1
import numpy as np
import cv2
from google.cloud import aiplatform

app = Flask(__name__)

def detect_clothing(image_content):
    client = vision_v1.ImageAnnotatorClient()
    image = vision_v1.Image(content=image_content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    # Convert bytes to OpenCV image
    image_cv = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = image_cv.shape

    # Define a list of fashion-related labels
    fashion_labels = ['Shirt', 'Dress', 'Shoe', 'Hat', 'Jacket', 'Skirt', 'Pants', 'Tie', 'Scarf']

    # List to store object details
    object_details = []

    # Extract and save fashion-related objects
    for object_ in objects:
        if object_.name in fashion_labels:
            vertices = [(int(vertex.x * width), int(vertex.y * height)) for vertex in object_.bounding_poly.normalized_vertices]
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(np.array(vertices))
            scaled_rect_x, scaled_rect_y, scaled_rect_w, scaled_rect_h = rect_x / width, rect_y / height, rect_w / width, rect_h / height
            cut_region = image_cv[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cut_region, object_.name, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cut_image_path = f'{object_.name}.jpg'
            cv2.imwrite(cut_image_path, cut_region)
            object_details.append({
                'id': cut_image_path.split('.')[0],
                'label': object_.name,
                'bounding_box': {'x': scaled_rect_x, 'y': scaled_rect_y, 'w': scaled_rect_w, 'h': scaled_rect_h},
                'image_path': cut_image_path
            })

    return object_details

def predict_with_vertex_ai(endpoint_id, instance, image_cv):
    aiplatform.init(project="6173592863218073600", location="us-central1")
    endpoint = aiplatform.Endpoint(endpoint_id)
    predictions = endpoint.predict(instances=[instance])

    # Assuming the predictions include bounding box coordinates and labels
    object_details = []
    for prediction in predictions.predictions:
        label = prediction['label']
        bounding_box = prediction['bounding_box']
        
        # Extract coordinates from bounding box
        x, y, w, h = bounding_box['x'], bounding_box['y'], bounding_box['w'], bounding_box['h']

        # Cut the region inside the bounding box
        cut_region = image_cv[y:y + h, x:x + w]

        # Add the label to the cut region
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cut_region, label, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Save the cut region as a new image
        cut_image_path = f'{label}.jpg'
        cv2.imwrite(cut_image_path, cut_region)

        # Add object details to the list
        object_details.append({
            'id': cut_image_path.split('.')[0],
            'label': label,
            'bounding_box': bounding_box,
            'image_path': cut_image_path
        })

    return object_details

@app.route('/')
def home():
    return "App is running!"

@app.route('/detect_clothing', methods=['POST'])
def detect_clothing_endpoint():
    image_file = request.files['image']
    image_content = image_file.read()
    
    # Convert bytes to OpenCV image
    image_cv = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)

    # Call the existing detect_clothing function for Vision API
    vision_api_details = detect_clothing(image_content)

    # Prepare the instance for Vertex AI (adjust as needed)
    vertex_ai_instance = {"image": image_content} # Adjust as needed

    # Call the Vertex AI prediction function
    vertex_ai_details = predict_with_vertex_ai("6173592863218073600", vertex_ai_instance, image_cv)

    # Combine or handle the results as needed
    result = {
        "vision_api_details": vision_api_details,
        "vertex_ai_details": vertex_ai_details
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
