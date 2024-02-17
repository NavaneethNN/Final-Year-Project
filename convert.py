import cv2
import torch
import numpy as np
import sys
sys.path.insert(0, './yolov5')



def preprocess_image(image):
    resized_image = cv2.resize(image, (640, 640))
    normalized_image = resized_image.astype(np.float32) / 255.0
    transposed_image = normalized_image.transpose((2, 0, 1))
    batched_image = np.expand_dims(transposed_image, axis=0)
    return batched_image

def postprocess_outputs(outputs, confidence_threshold):
    boxes, scores, class_ids = [], [], []
    for det in outputs:
        box = det[0:4]
        score = det[4]
        class_id = int(det[5])

        if score > confidence_threshold:
            boxes.append(box)
            scores.append(score)
            class_ids.append(class_id)

    return boxes, scores, class_ids

def draw_detections(image, boxes, scores, labels, colors):
    opencv_boxes = [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) for box in boxes]

    for (left, top, right, bottom), score, label, color in zip(opencv_boxes, scores, labels, colors):
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(image, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# Load your YOLOv5 model from .pt file
model = torch.load("D:/project/yolov5/runs/train/exp2/weights/best.pt")['model'].float()


model.eval()

# Capture video
cap = cv2.VideoCapture(0)

# Define colors for each class
colors = [(0, 255, 0), (255, 0, 0)]  # Customize the colors based on your classes

while True:
    ret, frame = cap.read()

    # Preprocess the frame
    preprocessed_image = preprocess_image(frame)

    # Run inference
    with torch.no_grad():
        outputs = model(torch.from_numpy(preprocessed_image))

    # Postprocess the outputs
    confidence_threshold = 0.5  # You can adjust this threshold
    boxes, scores, class_ids = postprocess_outputs(outputs[0], confidence_threshold)

    # Save the annotated frame to an image file
    annotated_frame = draw_detections(frame.copy(), boxes, scores, class_ids, colors)
    cv2.imshow('Real-time Object Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
