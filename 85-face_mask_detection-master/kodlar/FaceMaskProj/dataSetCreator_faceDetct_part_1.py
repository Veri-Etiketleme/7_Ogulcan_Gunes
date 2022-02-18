import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
id = input('enter user Id:')
sampleNo = 0
while (True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        sampleNo = sampleNo +1
        cv2.imwrite('C:/Users/KHAAN/OneDrive/Desktop/dataset/without_Mask' +str(id)+'.'+str(sampleNo)+".jpg",gray[y: y+h, x:x+w])
        cv2.putText(img, str(sampleNo), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img,(x,y), (x+w, y+h), (0,0,255),2)
    cv2.imshow('faces are :', img)
    if cv2.waitKey(10) == 13 or sampleNo == 250:
        break
cam.release()
cv2.destroyAllWindows()
print('sample complete:')