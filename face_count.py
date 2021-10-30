import cv2
import numpy as np
import dlib as d

cap = cv2.VideoCapture(0)

detector = d.get_frontal_face_detector()

while True:
    sus,video = cap.read()
    video = cv2.flip(video,1)
    gray = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    i = 0

    for face in faces:
        x,y = face.left(), face.top()
        x1,y1 = face.right(), face.bottom()
        cv2.rectangle(video,(x,y),(x1,y1),(0,255,0),0)

    cv2.imshow("videos frame",video)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()    