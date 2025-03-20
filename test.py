import onnxruntime as ort
import cv2
import numpy as np

# Load the ONNX model
model_path = "best.onnx"
session = ort.InferenceSession(model_path)

# Get input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # Example: (1, 3, 640, 640)
input_height, input_width = input_shape[2], input_shape[3]

# Load and preprocess the image
image_path = r"C:\Users\begat\Downloads\in1.jpg"  # Use raw string
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
resized_image = cv2.resize(image, (input_width, input_height))  # Resize to model input size
input_image = resized_image.transpose(2, 0, 1)  # Change from HWC to CHW format
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
input_image = input_image.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Run inference
outputs = session.run(None, {input_name: input_image})

# Process outputs (assuming YOLOv8 output format)
output = outputs[0]  # Shape: [1, 84, 8400]
output = np.squeeze(output)  # Remove batch dimension: [84, 8400]

# Extract bounding boxes, scores, and class IDs
boxes = output[:4, :]  # First 4 rows: bounding box coordinates (x1, y1, x2, y2)
scores = output[4:, :].max(axis=0)  # Max score across classes
class_ids = output[4:, :].argmax(axis=0)  # Class ID with max score

# Filter detections by confidence score
confidence_threshold = 0.5
filtered_indices = scores > confidence_threshold
boxes = boxes[:, filtered_indices]
scores = scores[filtered_indices]
class_ids = class_ids[filtered_indices]

# Rescale bounding boxes to original image size
scale_x = image.shape[1] / input_width
scale_y = image.shape[0] / input_height
boxes[0, :] *= scale_x  # x1
boxes[1, :] *= scale_y  # y1
boxes[2, :] *= scale_x  # x2
boxes[3, :] *= scale_y  # y2

# Draw bounding boxes on the image
for i in range(boxes.shape[1]):
    x1, y1, x2, y2 = boxes[:, i].astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box
    label = f"Class {class_ids[i]}: {scores[i]:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow("ONNX Inference", image)
cv2.waitKey(0)
cv2.destroyAllWindows()