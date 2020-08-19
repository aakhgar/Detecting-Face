import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


videocapture = cv2.VideoCapture(0)

scale_factor = 1.3

while True:
    ret, pic = videocapture.read()
    faces = face_cascade.detectMultiScale(pic, scale_factor, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(pic, (x,y), (x+w, y+h), (255,0,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(pic, 'Alireza', (x,y), font, 2, (255,255,255), 2, cv2.LINE_AA)
        
    print("number of face found {}".format(len(faces)))
    cv2.imshow('faces', pic)
    k= cv2.waitKey(30) & 0xFF
    if k== 2:
        break
    
cv2.destroyallwindows()
