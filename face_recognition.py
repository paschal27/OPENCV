import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Pascal', 'Ben Afflek']
# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'Photos/B.jpg')
scale_percent = 0.2 # percent of original size
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)
dim = (width, height)
  
# resize image
resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)


cv.waitKey(0)
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
for(x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

label, confidence = face_recognizer.predict(faces_roi)
print(f'Label = {people[label]} with a confidence of {confidence}')

cv.putText(resized, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
cv.rectangle(resized, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', resized)

cv.waitKey(0)

# capture = cv.VideoCapture(0)
# while True:
#     isTrue, frame = capture.read()
#     cv.imshow('Video', frame)

#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
    
#     haar_cascade = cv.CascadeClassifier('haar_face.xml')

#     people = ['Pascal']
#     # features = np.load('features.npy')
#     # labels = np.load('labels.npy')

#     face_recognizer = cv.face.LBPHFaceRecognizer_create()
#     face_recognizer.read('face_trained.yml')
    
#     faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)
#     for(x,y,w,h) in faces_rect:
#         faces_roi = frame[y:y+h, x:x+w]

#     label, confidence = face_recognizer.predict(faces_roi)
#     print(f'Label = {people[label]} with a confidence of {confidence}')

#     cv.putText(frame, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
#     cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

#     cv.imshow('Detected Faces', frame)

# capture.release()
# cv.destroyAllWindows()