from inference_sdk import InferenceHTTPClient
import cv2
import sys
import json

# Get arguments from command line
image_path = sys.argv[1]

# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="sjTIrL3pv5ztvkDrMwTC"
)

# Verify image path and load using OpenCV
image = cv2.imread(image_path)

# Perform inference using the SDK
result = CLIENT.infer(image_path, model_id="eg-seg7z/1")

# Print result as JSON
print(json.dumps(result))
