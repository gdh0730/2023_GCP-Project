from flask import Flask, request, jsonify
from google.cloud import vision_v1, aiplatform
from google.cloud.aiplatform.gapic.schema import predict
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "App is running!"

@app.route('/detect_clothing_vision', methods=['POST'])
def detect_clothing_vision_endpoint():
    image_file = request.files['image']
    image_content = image_file.read()
    object_details = detect_clothing(image_content)
    return jsonify(object_details)

@app.route('/detect_clothing_vertex', methods=['POST'])
def detect_clothing_vertex_endpoint():
    image_file = request.files['image']
    image_content = image_file.read()
    # Save the image to a temporary file for processing
    image_path = "temp_image.jpeg"
    with open(image_path, "wb") as img_file:
        img_file.write(image_content)
    object_details = predict_image_object_detection_sample(
        project="568907669076",
        endpoint_id="6173592863218073600",
        location="us-central1",
        filename=image_path
    )
    os.remove(image_path)  # Clean up temporary image
    return jsonify(object_details)

def detect_clothing(image_content):
    client = vision_v1.ImageAnnotatorClient()
    image = vision_v1.Image(content=image_content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    # Convert bytes to OpenCV image
    image_cv = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = image_cv.shape

    # Define a list of fashion-related labels
    fashion_labels = ['Shirt', 'Dress', 'Shoe', 'Hat', 'Jacket', 'Skirt', 'Pants', 'Tie', 'Scarf','outerwear']

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

def predict_image_object_detection_sample(project, endpoint_id, filename, location="us-central1", api_endpoint="us-central1-aiplatform.googleapis.com"):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    
    with open(filename, "rb") as f:
        file_content = f.read()

    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageObjectDetectionPredictionInstance(content=encoded_content).to_value()
    instances = [instance]
    
    parameters = predict.params.ImageObjectDetectionPredictionParams(confidence_threshold=0.5, max_predictions=5).to_value()
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
    
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
    
    # Additional processing to extract the most confident prediction and return in desired format
    prediction_data = response.predictions[0]
    highest_confidence = max(prediction_data["confidences"])
    index_of_highest_confidence = prediction_data["confidences"].index(highest_confidence)
    bbox_highest_confidence = prediction_data["bboxes"][index_of_highest_confidence]
    name_highest_confidence = prediction_data["displayNames"][index_of_highest_confidence]

    # Create a dictionary to store the details of the most confident prediction
    most_confident_details = {
        'label': name_highest_confidence,
        'confidence': highest_confidence,
        'bounding_box': {
            'top_left': {'x': bbox_highest_confidence[0], 'y': bbox_highest_confidence[3]},
            'bottom_right': {'x': bbox_highest_confidence[1], 'y': bbox_highest_confidence[2]}
        }
    }

    return most_confident_details

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
