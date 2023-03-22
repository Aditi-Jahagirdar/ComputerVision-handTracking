import cv2
import mediapipe as mp
import time
import NewHandTrackingModule as nthm


pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
# detector is object of handDetector class created with default parameters
detector = nthm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:     #otherwise 0 will be index out of range if hand is not detected,list length is zero
        print(lmList[7])     #7 is index that represents tip of the thumb, will print value only for id7

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # To display fps on screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Optional -code for close with q button or press q to quit
        break
cap.release()
