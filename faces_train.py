import cv2
import os
import cv2 as cv
import numpy as np

people = ['Pascal', 'Ben Afflek']

DIR = r'C:\Users\onyek\Downloads\OPENCV\Faces'
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        # if person != people:
        #     label = []
        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            scale_percent = 0.6 # percent of original size
            width = int(img_array.shape[1] * scale_percent)
            height = int(img_array.shape[0] * scale_percent)
            dim = (width, height)
  
            # resize image
            resized = cv.resize(img_array, dim, interpolation = cv.INTER_AREA)
            gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for(x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
print('Training Done')
print(f'Length of features = {len(features)}')
print(f'Length of features = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#Train the recognizer

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
