import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
    #Initialization
        #creating object "self" that has its own variable mode and assigning it value "mode" , self.mode=mode
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # To detect the hands- create object "hands" related to "Hands" class
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectionCon, self.trackCon)
        #instead of default parameters, handDetector class object's variables are defined as parameters
        # method by mediapipe to draw line connecting points on hand (there are 21 such points on each hand)
        self.mpDraw = mp.solutions.drawing_utils

    #Detection
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    #Find position of landmarks in hands usinf findpostion method
    def findPosition (self, img,handNo = 0, draw = True):     #handNo to get information of specific hand number
        lmList = []  #landmark list will have all landmark position

        if self.results.multi_hand_landmarks:  #to check if any hands were detected or not
            myHand = self.results.multi_hand_landmarks[handNo]  #myHand Points to particular hand number,
            # self object can be used in all methods
            #To get all landmarks within that hand
            for id, lm in enumerate(myHand.landmark):
                #print(id,lm)
                h, w, c = img.shape  # h-height, w-width , c-channel
                cx, cy = int(lm.x * w), int(lm.y * h)  # position  of centre
                #print(id, cx, cy)
                lmList.append([id, cx, cy])   #append landmark to list
                if draw:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

        return lmList



def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    # detector is object of handDetector class created with default parameters
    detector = handDetector()

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
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Optional -code for close with q button or press q to quit
             break
    cap.release()

if __name__ == "__main__":
    main()
