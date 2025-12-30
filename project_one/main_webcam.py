from ultralytics import YOLO
import cv2 as cv
import cvzone as cvz
from time import sleep
from math import ceil

capture = cv.VideoCapture(0)
capture.set(3, 1280)    # 640, 480
capture.set(4, 720)


model = YOLO("../yolo-weights/yolov8s.pt")

object_class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]



while True:
    success, img = capture.read()


    results = model(img, stream=True)

    for result in results:
        # print(result)
        boxes = result.boxes

        for box in boxes:

            # bounding box display
            x1,y1, x2,y2 = box.xyxy[0]
            x1,y1, x2,y2 = int(x1),int(y1),int(x2),int(y2)  # cordinates = list(map(int, box.xyxy[0]))
            
            # cv.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            # # fancy rectangle
            # x,y, w,h = box.xywh[0]
            # x,y, w,h = int(x),int(y),int(w),int(h)


            # confidence display
            confidence = ceil(box.conf[0]*100)
            

            # type of object display
            object_class_index = int(box.cls[0])
            current_object_class = object_class_names[object_class_index]

            if current_object_class == "person" and confidence > 75:
                cvz.cornerRect(img, (x1,y1, (x2-x1),(y2-y1)))
                cvz.putTextRect(img, f"C:{confidence}% {current_object_class}", (max(0, x1),max(40, y1-10)), scale=1.5, thickness=1, offset=2)


    cv.imshow("webcam", img)
    # sleep(0.3)  # to stop it, from melting my cpu
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
# capture.release()
# cv.destroyAllWindows()