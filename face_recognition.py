import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')



id_object = ['Ajanta', 'Ravana Statue', 'Shiva Statue', 'Vaishnavi']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\avshe\Desktop\Open CV\face_trained_personal.yml')

img = cv.imread('C:\\Users\\avshe\\Desktop\\Vaish\\vaishnavi image.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {id_object[label]} with a confidence of {confidence}')

    cv.putText(img, str(id_object[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)