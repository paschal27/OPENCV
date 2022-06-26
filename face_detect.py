import cv2 as cv

img = cv.imread('Faces/Pascal/5.jpg')

scale_percent = 0.7 # percent of original size
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)
dim = (width, height)
  
# resize image
resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)

cv.imshow('Person', resized)

gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Person', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

print(f'Number of faces found = {len(faces_rect)}')

for(x,y,w,h) in faces_rect:
    cv.rectangle(resized, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', resized)
cv.waitKey(0)

# capture = cv.VideoCapture(0)

# while True:
#     isTrue, frame = capture.read()
#     cv.imshow('Video', frame)

#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break

#     haar_cascade = cv.CascadeClassifier('haar_face.xml')

#     faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=1)

#     print(f'Number of faces found = {len(faces_rect)}')

#     for(x,y,w,h) in faces_rect:
#         cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

#     cv.imshow('Detected Faces', frame)

# capture.release()
# cv.destroyAllWindows()