import cv2
import numpy as np

# показ картинки
# img = cv2.imread('ye.jpg')
# img = cv2.GaussianBlur(img, (9, 9), 0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.Canny(img, 100, 100)

# kernel = np.ones((5, 5), np.uint8)

# img = cv2.dilate(img, kernel, iterations=1)

# img = cv2.erode(img, kernel, iterations=1)

# NewImg = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

# cv2.imshow('yea', NewImg)

# print(img.shape )

# cv2.waitKey(-1)

#||||||||||
# from cv2_enumerate_cameras import enumerate_cameras

# for camera_info in enumerate_cameras():
#     print(f'{camera_info.index}: {camera_info.name}')
#||||||||||


#показ видео/ захват вебки
# cap = cv2.VideoCapture(700)
# cap.set(3, 1000)
# cap.set(4, 1000)

# while True:
#     success, img = cap.read()

#     img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.Canny(img, 100, 100)
#     kernel = np.ones((5,5), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=1)
#     img = cv2.erode(img, kernel, iterations=1)

#     cv2.imshow('goofyy aah', img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#||||||||||||||||||||||||||||||||||||||||||||||||||


#СОЗДАНИЕ фото
# photo = np.zeros((1000,1000, 3), dtype='uint8')

# photo[0:150, 50:100] =204, 192, 255

# cv2.rectangle(photo, (93,123), (834, 234), (204, 192, 255), thickness=cv2.FILLED)

# cv2.line(photo, (0, photo.shape[1]//2), (photo.shape[0], photo.shape[1]//2), (255, 255, 255), thickness=3)

# cv2.circle(photo, (photo.shape[1]//2, photo.shape[0]//2), 200, (255,255,255), thickness=cv2.FILLED)

# cv2.putText(photo, 'Hello world!', (photo.shape[1]//2-86, photo.shape[0]//2+250), cv2.FONT_ITALIC, 1, (255,255,255), thickness=1)

# cv2.imshow('idk', photo)

# cv2.waitKey(0)
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#ротация и перемещение
# img = cv2.imread('ye.jpg')
# img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))




# def rotate(img, angle: int):
#     height, width = img.shape[:2]
#     point = (width//2, height//2)

#     mat = cv2.getRotationMatrix2D(point, angle, 1)
#     return cv2.warpAffine(img, mat, (width, height))

# def transform(img, x, y):
#     mat = np.float32([[1,0,x],[0,1,y]])
#     return cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))


# img = transform(img, 30, 200)
# cv2.imshow('ye', img)

# cv2.waitKey(0)
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#создание копии контура
# img = cv2.imread('ye.jpg')

# NewImg = np.zeros(img.shape, dtype='uint8')


# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img, (5,5), 0)

# img = cv2.Canny(img, 100, 100)

# contour, hirearhy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# cv2.drawContours(NewImg, contour, -1, (255,255,255), 1)

# cv2.imshow('ye', NewImg)
# cv2.waitKey(0)
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#разделение на r,g,b
# img = cv2.imread('ye.jpg')

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# r, g, b = cv2.split(img)

# img = cv2.merge([b,g,r])




# cv2.imshow('yuppy', img)
# cv2.waitKey(0)
#||||||||||||||||||||||||||||||||||||||||

#создание маски
# photo = cv2.imread('ye.jpg')
# img = np.zeros(photo.shape[:2], dtype='uint8')

# circle = cv2.circle(img.copy(), (600, 600), 900, 255, cv2.FILLED)
# square = cv2.rectangle(img.copy(), (300, 300), (500, 500), 255, -1)

# img = cv2.bitwise_and(photo, photo, mask=circle)

# cv2.imshow('ye', img)


# cv2.waitKey(0)
#||||||||||||||||||||||||||||||||||||||||||||||


#распознавание лиц
# img = cv2.imread('ppl3.jpeg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = cv2.CascadeClassifier('faces.xml')

# results = faces.detectMultiScale(gray, scaleFactor=2, minNeighbors=3)

# for (x, y, w, h) in results:
#     cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), thickness=1)



# cv2.imshow('a', img)
# cv2.waitKey(0)
#|||||||||||||||||||||||||||||||||||||||||||||||||||

cap = cv2.VideoCapture(700)
cap.set(3, 1000)
cap.set(4, 1000)

while True:
    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cv2.CascadeClassifier('faces.xml')

    results = faces.detectMultiScale(gray, scaleFactor=2, minNeighbors=3)

    for (x, y, w, h) in results:
        cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv2.imshow('goofyy aah', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break