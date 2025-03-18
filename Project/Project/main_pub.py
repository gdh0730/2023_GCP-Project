from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision_v1
from google.cloud import storage
import numpy as np
import json
import cv2
from io import BytesIO
from serpapi import GoogleSearch
from google.cloud import pubsub_v1

app = Flask(__name__)
CORS(app)

# Initialize GCS client
storage_client = storage.Client()
bucket_name = "fashion-dataset-storage"  # Replace with your bucket name
bucket = storage_client.bucket(bucket_name)

# Initialize the Pub/Sub publisher client
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('jnu-idv-03', 'pubtobigtopic')

API_KEY = ""  # Replace with your SerpAPI key

@app.route('/')
def home():
    return "App is running!"

@app.route('/detect_clothing', methods=['POST'])
def detect_clothing_endpoint():
    image_file = request.files['image']
    image_content = image_file.read()
    detect_clothing(image_content)
    return jsonify({"message": "Data published to Pub/Sub successfully!"})

def detect_clothing(image_content):
    client = vision_v1.ImageAnnotatorClient()
    image = vision_v1.Image(content=image_content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    # Convert bytes to OpenCV image
    image_cv = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = image_cv.shape

    fashion_labels = ['Person','Outerwear','Top','Sleeve','Jacket','Suit','Pants','Dress','Vest']

    # all_visual_matches = []

    for object_ in objects:
        if object_.name in fashion_labels:
            vertices = [(int(vertex.x * width), int(vertex.y * height)) for vertex in object_.bounding_poly.normalized_vertices]
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(np.array(vertices))
            cut_region = image_cv[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]

            # Convert the cropped image to byte stream
            is_success, buffer = cv2.imencode(".jpg", cut_region)
            byte_stream = BytesIO(buffer)

            # Upload to GCS
            blob_path = f'cropped_images/{object_.name}_{rect_x}_{rect_y}.jpg'
            blob = bucket.blob(blob_path)
            blob.upload_from_file(byte_stream, content_type="image/jpeg")

            # Use SerpAPI to search for the image
            image_url = f"https://storage.googleapis.com/{bucket_name}/{blob_path}"
            params = {
                "api_key": API_KEY,
                "engine": "google_lens",
                "url": image_url
            }
            search = GoogleSearch(params)
            serp_results = search.get_dict()

            visual_matches = serp_results.get("visual_matches", [])
            for match in visual_matches:
                extracted_data = {
                    "title": match.get("title", ""),
                    "link": match.get("link", ""),
                    "thumbnail": match.get("thumbnail", ""),
                    "price":match.get("price","")
                }
                # all_visual_matches.append(extracted_data)
                data_str = json.dumps(extracted_data)  # Convert the data to a JSON string
                publisher.publish(topic_path, data=data_str.encode('utf-8'))

            # Delete the image from GCS after searching with SerpAPI
            blob_to_delete = bucket.blob(blob_path)
            blob_to_delete.delete()

    # return all_visual_matches


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
