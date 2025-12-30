from ultralytics import YOLO
import cv2 as cv
import cvzone as cvz
from numpy import empty, array, vstack
from time import sleep
from math import ceil
from sort import *


# capture = cv.VideoCapture(0)
# capture.set(3, 1280)    # 640, 480
# capture.set(4, 720)

capture = cv.VideoCapture("../videos/light_traffic.mp4")

model = YOLO("../yolo-weights/yolov8n.pt")

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

valid_vehicle_highway = [
    "car", "motorbike", "bus", "truck"
]

mask = cv.imread("../masks/heavy_traffic_mask.png")

tracker = Sort(max_age=24, min_hits=3, iou_threshold=0.5)

line_one = [230,300,500,300]
line_two = [555,360,705,360]
line_three = [915,470,1190,470]

total_car_count_dict = {}
total_car_count = []
line_cross = lambda cordinates_list: cordinates_list[0] < cx < cordinates_list[2] and cordinates_list[1]-30 < cy < cordinates_list[3]+30



while True:
    success, img = capture.read()
    img_region = cv.bitwise_and(img, mask)

    results = model(img_region, stream=True)

    detections = empty((0, 5))

    cv.line(img, (line_one[0],line_one[1]), (line_one[2],line_one[3]), (200,2,2), 5)
    cv.line(img, (line_two[0],line_two[1]), (line_two[2],line_two[3]), (200,2,2), 5)
    cv.line(img, (line_three[0],line_three[1]), (line_three[2],line_three[3]), (200,2,2), 5)

    for result in results:
        # print(result)
        boxes = result.boxes

        for box in boxes:

            x1,y1, x2,y2 = box.xyxy[0]
            x1,y1, w, h = int(x1),int(y1), int(x2-x1), int(y2-y1)  # cordinates = list(map(int, box.xyxy[0]))
            # w, h = (x2-x1), (y2-y1)

            # cv.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            # # fancy rectangle
            # x,y, w,h = box.xywh[0]
            # x,y, w,h = int(x),int(y),int(w),int(h)

            # bounding box display

            # confidence display
            confidence = ceil(box.conf[0]*100)
            # print(confidence)

            # type of object display
            object_class_index = int(box.cls[0])
            current_object_class = object_class_names[object_class_index]

            if current_object_class in valid_vehicle_highway and (confidence) > 40:

                # cvz.cornerRect(img, (x1,y1, w, h), l=8, t=1, rt=5)
                # cvz.putTextRect(img, f"C:{confidence}% {current_object_class}", (max(0, x1), max(40, y1-10)), scale=1, thickness=1, offset=1)

                current_array = array([x1,y1, x2,y2, confidence])
                detections = vstack((detections, current_array))


    tracked_results = tracker.update(detections)

    

    # 2:04:0
    
    for result in tracked_results:
        x1,y1, x2,y2, id = result
        # print(result)
        x1, y1, w, h, id= int(x1), int(y1), int(x2-x1), int(y2-y1), str(id)
       

        # cvz.cornerRect(img, (x1,y1, w, h), l=10, t=1, rt=5, colorR=(0,255,0))
        # cvz.putTextRect(img, f"{id}", (max(0, x1), max(40, y1-10)), scale=1, thickness=1, offset=1)

        cx, cy = x1+w//2, y1+h//2
        # cv.circle(img, (cx,cy), 5, (255,0,0), cv.FILLED)

        

        if line_cross(cordinates_list=line_one) or line_cross(cordinates_list=line_two) or line_cross(cordinates_list=line_three):
            
            if id not in total_car_count_dict:
                total_car_count_dict[id] = 1
            cvz.putTextRect(img, f"Detected", (450,40), colorR=(0,155,0))
                

            # if total_car_count.count(id) == 0:
            #     total_car_count.append(id)
    

    cvz.putTextRect(img, f"Cars Counted: {len(total_car_count_dict)}", (5,38), colorR=(0,0,0))


    cv.imshow("CCTV Intersection", img)
    # cv.imshow("Video Mask", img_region)
    # sleep(0.3)  # to stop it, from melting my cpu
    if cv.waitKey(0) & 0xFF == ord('q'):
        break
    
# capture.release()
# cv.destroyAllWindows()