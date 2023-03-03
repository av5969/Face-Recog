import os
import cv2 as cv
import numpy as np

id_object = ['Ajanta', 'Ravana Statue', 'Shiva Statue', 'Vaishnavi']
DIR = r'C:\Users\avshe\Desktop\Open CV\Faces\Train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')
haar_cascade = cv.CascadeClassifier('haar_fullbody.xml')
haar_cascade = cv.CascadeClassifier('haar_lowerbody.xml')
haar_cascade = cv.CascadeClassifier('haar_upperbody.xml')
features = []
labels = []

def create_train():
    for person in id_object:
        path = os.path.join(DIR, person)
        label = id_object.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            # Laplacian
            lap = cv.Laplacian(gray, cv.CV_64F)
            lap = np.uint8(np.absolute(lap))


            # Sobel 
            sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
            sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
            combined_sobel = cv.bitwise_or(sobelx, sobely)



            canny = cv.Canny(gray, 150, 175)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)