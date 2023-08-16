import cv2

face = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(face)

#uses main camera
webcam = cv2.VideoCapture(0)

while True:
    _, image = webcam.read()
    #converts image into greyscale and runs detection
    greyimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(greyimage, 1.3, 3)

    #display detected face inside rectangle
    for (x,y,w,h) in face:
        cv2.rectangle(image,(x,y),(x+w,y+h),(200,200,0),2)

    #opens display window
    cv2.imshow("Face Detector",image)

    #stops program after pressing esc
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()