import cv2
from yolo_segmentation import YOLOSegmentation

img = cv2.imread("../images-extracted/image12_0.png")
img = cv2.resize(img, None, fx=0.7, fy=0.7)

ys = YOLOSegmentation("yolov8m-seg.pt")
bboxes, class_id, segmentations, scores = ys.detect(img)
print(bboxes)

cv2.imshow("image", img)
cv2.waitKey(0)