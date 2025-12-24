from ultralytics import YOLO
import cv2 as cv
import cvzone as cvz
from time import sleep
from math import ceil

capture = cv.VideoCapture(0)
capture.set(3, 1280)    # 640, 480
capture.set(4, 720)


model = YOLO("../yolo-weights/yolov8n.pt")


while True:
    success, img = capture.read()


    results = model(img, stream=True)

    for result in results:
        # print(result)
        boxes = result.boxes

        for box in boxes:
            
            # boring rectangle

            x1,y1, x2,y2 = box.xyxy[0]
            x1,y1, x2,y2 = int(x1),int(y1),int(x2),int(y2)  # cordinates = list(map(int, box.xyxy[0]))
            
            # cv.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
       
            # # fancy rectangle
            # x,y, w,h = box.xywh[0]
            # x,y, w,h = int(x),int(y),int(w),int(h)

            cvz.cornerRect(img, (x1,y1, (x2-x1),(y2-y1)))

            # 53:00
            confidence = ceil(box.conf[0]*100)/100 
            print(confidence)
            cvz.putTextRect(img, f"{confidence}", (x1,y1-10))


    cv.imshow("webcam", img)
    sleep(0.3)  # to stop it, from melting my cpu
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
# capture.release()
# cv.destroyAllWindows()