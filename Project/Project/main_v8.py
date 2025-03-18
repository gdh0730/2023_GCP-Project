from flask import Flask, request, jsonify
import os
import cv2
from google.cloud import aiplatform, vision_v1, storage
from google.cloud.aiplatform.gapic.schema import predict
from io import BytesIO
import base64
import numpy as np

app = Flask(__name__)

# GCS setup
storage_client = storage.Client()
bucket_name = "fashion-dataset-storage" 
bucket = storage_client.bucket(bucket_name)

def save_to_gcs(buffer, path):
    byte_stream = BytesIO(buffer)
    blob = bucket.blob(path)
    blob.upload_from_file(byte_stream, content_type="image/jpeg")
    return f'gs://{bucket_name}/{path}'

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
    return response.predictions[0]

def detect_clothing(image_content):
    client = vision_v1.ImageAnnotatorClient()
    image = vision_v1.Image(content=image_content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    image_cv = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = image_cv.shape

    fashion_labels = ['Shirt', 'Dress', 'Shoe', 'Hat', 'Jacket', 'Skirt', 'Pants', 'Tie', 'Scarf','outerwear']

    for object_ in objects:
        if object_.name in fashion_labels:
            vertices = [(int(vertex.x * width), int(vertex.y * height)) for vertex in object_.bounding_poly.normalized_vertices]
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(np.array(vertices))
            cut_region = image_cv[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]

            is_success, buffer = cv2.imencode(".jpg", cut_region)
            if is_success:
                blob_path = f'cropped_images/vision_{object_.name}_{rect_x}_{rect_y}.jpg'
                save_to_gcs(buffer, blob_path)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "Image not provided"}), 400

    image_content = image_file.read()

     # Vision API Predictions
    detect_clothing(image_content)
    
    # AI Platform Predictions
    image_content = image_file.read()
    image_path = os.path.join('/tmp', image_file.filename)
    image_file.save(image_path)

    aiplatform_predictions = predict_image_object_detection_sample(
        project="568907669076",
        endpoint_id="6173592863218073600",
        location="us-central1",
        filename=image_path
    )

    prediction_data = aiplatform_predictions[0]
    highest_confidence = max(prediction_data["confidences"])
    index_of_highest_confidence = prediction_data["confidences"].index(highest_confidence)
    bbox_highest_confidence = prediction_data["bboxes"][index_of_highest_confidence]
    name_highest_confidence = prediction_data["displayNames"][index_of_highest_confidence]

    most_confident_details = {
        'label': name_highest_confidence,
        'confidence': highest_confidence,
        'bounding_box': {
            'top_left': {'x': bbox_highest_confidence[0], 'y': bbox_highest_confidence[3]},
            'bottom_right': {'x': bbox_highest_confidence[1], 'y': bbox_highest_confidence[2]}
        }
    }
    
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    top_left_x = int(most_confident_details["bounding_box"]["top_left"]["x"] * width)
    top_left_y = int(most_confident_details["bounding_box"]["top_left"]["y"] * height)
    bottom_right_x = int(most_confident_details["bounding_box"]["bottom_right"]["x"] * width)
    bottom_right_y = int(most_confident_details["bounding_box"]["bottom_right"]["y"] * height)
    roi = image[bottom_right_y:top_left_y, top_left_x:bottom_right_x]

    is_success, buffer = cv2.imencode(".jpg", roi)
    if is_success:
        aiplatform_image_path = f'cropped_images/aiplatform_cropped_{image_file.filename}'
        save_to_gcs(buffer, aiplatform_image_path)

    # Cleanup
    os.remove(image_path)

    return "Predictions processed and images saved."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
