#pip install opencv-python
import cv2
import time

face_cascade = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier("cascade/haarcascade_smile.xml")
counter = 1
smile_detected = False
start_time = 0

#uses main camera
webcam = cv2.VideoCapture(0)

while True:
    _, image = webcam.read()
    #converts image into greyscale and runs detection
    greyimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(greyimage, 1.3, 3)
    for (x,y,w,h) in face:
        bound_gray = greyimage[y:y+h, x:x+w]
        bound_color = image[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(bound_gray,1.8,20) 
        #takes selfie if smile is detected and saves to local folder
        if not smile_detected and len(smile)>0:
            smile_detected = True
            start_time = time.time()
        elif not len(smile)>0:
            smile_detected = False
        elif smile_detected and time.time() - start_time >= 1:
            filename = "selfie{}.jpg".format(counter)
            cv2.imwrite(filename,image)
            print("Photo Captured")
            counter += 1
            start_time = time.time()

    #opens display window
    cv2.imshow("Face Detector",image) 

    #stops program after pressing esc
    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()