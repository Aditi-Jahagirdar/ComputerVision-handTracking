import cv2
import mediapipe as mp
import time #to check the frame rate

#To create video object (with webcam 1)
cap = cv2.VideoCapture(0)

#To detect the hands- create object "hands" related to "Hands" class
mpHands = mp.solutions.hands
hands = mpHands.Hands()    #() empty parenthesis indicates default parameters for the object creation.
#method by mediapipe to draw line connecting points on hand (there are 21 such points on each hand)
mpDraw = mp.solutions.drawing_utils

#To calculate framerate (frame per second , fps = 1/(current - previous time))
pTime = 0  #previous time
cTime = 0 #current time

#To run a webcam
while True:
    success, img = cap.read()

#Convert  webcam image loaded by openCV(i.e. in BGR,openCV uses BGR format) into RGB image before sending
#Because Hands class uses only RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  #process method in hands object will process the results

    #print(results)  #prints nothing "<class 'mediapipe.python.solution_base.SolutionOutputs'>"
    print(results.multi_hand_landmarks)   #multi_hand_landmarks to check if hand is detected in results, prints x,y,z landmark otherwise prints "none"

    #To extract information of each hand using loop
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:   #where handLms represents each hand
            #To get ID and landmark information of each hand
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                #To convert decimal landmarks into pixels
                h, w, c = img.shape #h-height, w-width , c-channel
                cx , cy = int(lm.x*w), int(lm.y*h) #position  of centre
                print(id, cx, cy)

                if id == 4:   #4 is ID for tip of fingers
                    cv2.circle(img, (cx,cy), 15, (0, 255, 255), cv2.FILLED)    #15 size of circe (rad/dia?) , colour


            mpDraw.draw_landmarks(img, handLms , mpHands.HAND_CONNECTIONS)  #points are drawn on original image "img" because original image will be displayed not RGB image
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    #To display fps on screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    #where (10,70) coordinates where fps will be shown, then font type, scale,colour(coordinate representation),thickness


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): #Optional -code for close with q button or press q to quit
        break
cap.release()








