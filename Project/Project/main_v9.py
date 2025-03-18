from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.cloud import storage
from io import BytesIO
import base64
from serpapi import GoogleSearch

app = Flask(__name__)
CORS(app)

storage_client = storage.Client()
bucket_name = "fashion-dataset-storage"  # Replace with your bucket name
bucket = storage_client.bucket(bucket_name)
API_KEY = ""

@app.route('/')
def home():
    return "App is running!"

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
    return response.predictions

@app.route('/predict', methods=['POST'])
def get_prediction():
    # Get image from POST request
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "Image not provided"}), 400

    image_path = os.path.join('/tmp', image_file.filename)
    image_file.save(image_path)

    predictions = predict_image_object_detection_sample(
        project="568907669076",
        endpoint_id="6173592863218073600",
        location="us-central1",
        filename=image_path
    )

    prediction_data = predictions[0]
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

    # Encode the cropped image to JPEG format
    is_success, buffer = cv2.imencode(".jpg", roi)
    if not is_success:
        return jsonify({"error": "Failed to encode the cropped image"}), 500

    # Convert buffer to a byte stream
    byte_stream = BytesIO(buffer)

    # Upload byte stream to GCS
    destination_blob_name = f"cropped_images/cropped_{image_file.filename}"
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(byte_stream, content_type="image/jpeg")

    # Use SerpAPI to search information about the cropped image
    image_url = f"https://storage.googleapis.com/{bucket_name}/{destination_blob_name}"
    params = {
        "api_key": API_KEY,
        "engine": "google_lens",
        "url": image_url
    }
    search = GoogleSearch(params)
    serp_results = search.get_dict()

    # # Combine both AI Platform predictions and SerpAPI results in the response
    # response_data = {
    #     "ai_platform": most_confident_details,
    #     "serp_api": serp_results
    # }
    all_visual_matches = []
    visual_matches = serp_results.get("visual_matches", [])
        for match in visual_matches:
            extracted_data = {
                "title": match.get("title", ""),
                "link": match.get("link", ""),
                "thumbnail": match.get("thumbnail", ""),
                "price":match.get("price","")
            }
            all_visual_matches.append(extracted_data)

        # Delete the image from GCS after searching with SerpAPI
        blob_to_delete = bucket.blob(blob_path)
        blob_to_delete.delete()

    # Clean up the original image file
    os.remove(image_path)

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
