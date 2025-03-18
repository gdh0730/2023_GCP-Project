from flask import Flask, request, jsonify
from google.cloud import vision_v1
from google.cloud import storage
import numpy as np
import cv2
from io import BytesIO

app = Flask(__name__)

# Initialize GCS client
storage_client = storage.Client()
bucket_name = "fashion-dataset-storage"  # Replace with your bucket name
bucket = storage_client.bucket(bucket_name)

@app.route('/')
def home():
    return "App is running!"

@app.route('/detect_clothing', methods=['POST'])
def detect_clothing_endpoint():
    image_file = request.files['image']
    image_content = image_file.read()
    object_details = detect_clothing(image_content)
    return jsonify(object_details)

def detect_clothing(image_content):
    client = vision_v1.ImageAnnotatorClient()
    image = vision_v1.Image(content=image_content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    # Convert bytes to OpenCV image
    image_cv = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = image_cv.shape

    fashion_labels = ['Shirt', 'Dress', 'Shoe', 'Hat', 'Jacket', 'Skirt', 'Pants', 'Tie', 'Scarf','outerwear']

    object_details = []

    for object_ in objects:
        if object_.name in fashion_labels:
            vertices = [(int(vertex.x * width), int(vertex.y * height)) for vertex in object_.bounding_poly.normalized_vertices]
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(np.array(vertices))
            cut_region = image_cv[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cut_region, object_.name, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Convert the cropped image to byte stream
            is_success, buffer = cv2.imencode(".jpg", cut_region)
            byte_stream = BytesIO(buffer)

            # Upload to GCS
            blob_path = f'cropped_images/{object_.name}_{rect_x}_{rect_y}.jpg'
            blob = bucket.blob(blob_path)
            blob.upload_from_file(byte_stream, content_type="image/jpeg")

            object_details.append({
                'id': blob_path.split('/')[-1].split('.')[0],
                'label': object_.name,
                'bounding_box': {'x': rect_x / width, 'y': rect_y / height, 'w': rect_w / width, 'h': rect_h / height},
                'image_path': f'gs://{bucket_name}/{blob_path}'
            })

    return object_details

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
