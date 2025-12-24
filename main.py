from ultralytics import YOLO
import cv2 as cv
from time import sleep

# 
model = YOLO("../yolo-weights/yolov8n.pt")

results = model("images/04.png", show=True)
# print(result)
img = results[0].plot()

cv.imshow("Objects", img)
cv.waitKey(0)          
cv.destroyAllWindows()