from flask import Flask, request, jsonify
import json
import cv2
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    # Read the image file from the request
    file = request.files['image']
    image_path = f"./{file.filename}"
    file.save(image_path)

    # Initialize the client
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="sjTIrL3pv5ztvkDrMwTC"
    )

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Unable to load image"}), 400

    # Perform inference using the SDK
    try:
        result = client.infer(image_path, model_id="eg-seg7z/1")
        return jsonify(result)
        

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
