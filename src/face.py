import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('../cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle","rb") as f:
    labels = pickle.load(f)
    inverted_labels = {v:k for k,v in labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    #Capture Frames
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        #Region of Interest
        #print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        #recognize - model
        id,conf = recognizer.predict(roi_gray)
        if conf >=45:#and conf <=85:
            print(inverted_labels[id])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = inverted_labels[id]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item,roi_gray)

        #draw rectangle
        color = (255,0,0)  #BGR 0-255 
        stroke = 2
        end_cord_x = x + w
        end_cord_x = y + h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_x),color,stroke)
    #Display the result fram
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
    	break

#When done release the capture
cap.release()
cv2.destroyAllWindows()
