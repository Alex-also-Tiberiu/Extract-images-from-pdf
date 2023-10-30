from ultralytics import YOLO
import numpy

#Load the model
model = YOLO("weights/yolov8n-cls.pt")

# train the model
model.train(data='/Users/davidnyarko/Documents/KOBY/KobyGitHub/yolov8-silva/datasets/animals', epochs=100)

#Predict on a image, setting save to true it acctually draws the box and save the image
detection_output = model.predict(source="images-extracted/cat.jpg", conf=0.25, save=True)

#Display tensor array
print(detection_output)

#Display numpay array
print(detection_output[0].numpy())