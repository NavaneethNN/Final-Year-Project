import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import cv2
import numpy as np
import sys
sys.path.insert(0, './yolov5')
import pathlib
import matplotlib.pyplot as plt


# Use the appropriate path separator based on the operating system
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# Load your custom PyTorch model
model_path = "D:/project/best.pt"
model = torch.load(model_path, map_location=torch.device('cpu'))['model'].float()
model.eval()

# Load class labels for your custom model
with open("labels.txt", "r") as f:
    classes = [line.strip() for line in f]

# Open video capture
cap = cv2.VideoCapture(0)  # Use the correct video source index or file path

plt.ion()  # Turn on interactive mode for Matplotlib

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Preprocess the frame (if needed)
    # Example: You may need to resize, normalize, and transpose the frame
    input_data = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(input_data)

    # Access the last dimension of the tensor
    output_tensor = outputs[0][:, :, -1]

    # Visualize the results on the frame (if needed)
    # Example: Draw bounding boxes, labels, and confidence scores on the frame
    for detection in output_tensor.view(-1):  # Flatten the tensor
        print("Detection Confidence:", detection.item())
        if detection.item() > 0.5:
            # Get the bounding box coordinates from your detection results
            # Replace this with your actual logic for getting bounding box coordinates
            box = [0, 0, 100, 100]

            # Draw bounding box on the frame
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Display the resulting frame using Matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Real-time Object Detection')
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)
    plt.gcf().canvas.flush_events()  # Handle key events

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.ioff()  # Turn off interactive mode when done
plt.show()  # Display the final figure

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
