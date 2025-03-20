from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("best.pt")  # Ensure "best.pt" is in your working directory

# Load the image
source = r"C:\Users\begat\Downloads\in2.jpg"
image = cv2.imread(source)

# Run inference
results = model(source)

# Loop through detections
for result in results:
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert bounding box coordinates to integers
        cropped_object = image[y1:y2, x1:x2]  # Crop the detected region

        # Show the cropped image
        cv2.imshow(f"Detection {i+1}", cropped_object)

        # Save the cropped object (optional)
        cv2.imwrite(f"cropped_object_{i+1}.jpg", cropped_object)

cv2.waitKey(0)
cv2.destroyAllWindows()
