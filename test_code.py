import numpy as np
import cv2


#this is the cascade we just made. Call what you want
# chest_cascade = cv2.CascadeClassifier(r"G:\coding_folder\Python\Python3\jupyter\chest_cascade.xml")
# chest_cascade = cv2.CascadeClassifier("data/haarcascade/chest_cascade.xml")
# chest_cascade = cv2.CascadeClassifier(r"C:\Users\rimon\anaconda3\envs\pytorch_env\lib\site-packages\cv2\data\chest_cascade.xml")

# chest_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_chest.xml")
chest_cascade = cv2.CascadeClassifier(r"C:\Users\rimon\anaconda3\envs\cntk_env\lib\site-packages\cv2\data\haarcascade_chest_v2.xml")
# print("I'm in\n")


cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    chests = chest_cascade.detectMultiScale(gray, 1.3, 5)
    
    # image, reject levels level weights.
    
    for (x,y,w,h) in chests:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # roi_area = w*h

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()