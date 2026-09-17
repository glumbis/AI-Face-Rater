import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------- Put the path to the picture you want to rate here ----------
IMAGE_TO_RATE = "your_picture.jpg"

print("Take you foto with you face facing the camera, and dont make any grimaces.")

def take_selfie():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow("Selfie", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            selfieFrame = frame
            #cv2.imwrite("selfie.jpg", frame)
            
            cap.release()
            cv2.destroyAllWindows()

            return selfieFrame

#selfiePic = take_selfie()
selfiePic = cv2.imread(IMAGE_TO_RATE)

#punkter for aligning
i1 = 27
i2 = 8

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def landmark_detect(filename):
    xList = []
    yList = []

    img = filename

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 0:
        print("No face detected. Please take you foto with you face forward, and have a straight face.")
        time.sleep(7)
        quit()
    
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(image=gray, box=face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(img=img, center=(x, y), radius=5, color=(0, 255, 0), thickness=-1)
            xList.append(x)
            yList.append(y)
    return xList, yList


gender = input("Are you a boy or a girl? (write boy or girl) Answer here --> ")

selfieX, selfieY = landmark_detect(selfiePic)


def getPerfs():
    if gender.lower() == "boy":
        perfectX = [44, 49, 57, 62, 73, 97, 128, 162, 200, 239, 268, 293, 315, 324, 328, 332, 334, 61, 77, 103, 129, 154, 210, 238, 263, 289, 308, 186, 187, 187, 188, 164, 177, 191, 205, 218, 95, 112, 132, 149, 131, 111, 225, 240, 259, 276, 261, 242, 138, 157, 177, 194, 211, 230, 250, 231, 213, 195, 178, 157, 146, 178, 195, 212, 241, 211, 194, 177]
        perfectY = [248, 291, 334, 375, 414, 448, 476, 498, 502, 493, 467, 438, 404, 366, 325, 284, 242, 243, 226, 223, 227, 234, 231, 224, 220, 222, 236, 265, 292, 318, 346, 362, 365, 368, 363, 359, 269, 262, 262, 271, 275, 275, 269, 260, 260, 265, 271, 271, 407, 399, 395, 398, 394, 394, 400, 416, 425, 428, 428, 422, 408, 407, 407, 404, 402, 405, 408, 408]
    elif gender.lower() == "girl":
        perfectX = [123, 124, 130, 139, 154, 179, 207, 241, 280, 318, 351, 379, 402, 418, 428, 434, 437, 141, 167, 197, 224, 251, 317, 343, 370, 399, 424, 280, 280, 280, 279, 254, 266, 279, 292, 304, 171, 191, 216, 233, 210, 186, 328, 347, 372, 391, 374, 350, 217, 239, 261, 277, 293, 317, 339, 319, 296, 279, 260, 238, 229, 261, 278, 294, 327, 295, 278, 261]
        perfectY = [264, 307, 349, 391, 429, 462, 492, 517, 524, 518, 496, 467, 436, 399, 358, 315, 273, 216, 205, 208, 219, 234, 236, 223, 215, 214, 226, 271, 303, 335, 366, 381, 386, 392, 386, 380, 263, 251, 254, 274, 279, 276, 277, 259, 257, 269, 282, 283, 428, 418, 412, 416, 411, 417, 428, 453, 464, 466, 464, 453, 432, 434, 435, 433, 431, 435, 437, 435]
    else:
        print("Real beauty comes from the mind, therefor you are a fucking 0. Can't even write boy or girl... smh")
        time.sleep(7)
        quit()

    
    return perfectX, perfectY

def dist(x1, y1, x2, y2):
    return(np.sqrt((x1-x2)**2+(y1-y2)**2))

def movePoint(pointX, pointY, x, y):
    newX = pointX - x
    newY = pointY - y
    return newX, newY

def scalePoint(pointX, pointY, centerX, centerY, scalar):
    deltaX = pointX-centerX
    deltaY = pointY-centerY
    newX = centerX+scalar*deltaX
    newY = centerY+scalar*deltaY

    return newX, newY

def rotatePoint(pointX, pointY, centerX, centerY, angle):
    dX = pointX - centerX
    dY = pointY - centerY
    L = np.sqrt(dX**2+dY**2)
   
    oA = np.arctan2(dY, dX)

    nA = oA + angle
    newX = centerX + np.cos(nA)*L
    newY = centerY + np.sin(nA)*L
    return newX, newY

#cv2.imshow(winname="Face", mat=img)
def findFactors(selfieX, selfieY, perfectX, perfectY, pointIndex1 = i1, pointIndex2 = i2):
    
    
    x = selfieX[pointIndex1]-perfectX[pointIndex1]
    y = selfieY[pointIndex1]-perfectY[pointIndex1]

    scalar = dist(perfectX[pointIndex1], perfectY[pointIndex1], perfectX[pointIndex2], perfectY[pointIndex2])/dist(selfieX[pointIndex1], selfieY[pointIndex1], selfieX[pointIndex2], selfieY[pointIndex2])

    sdX = selfieX[pointIndex1] - selfieX[pointIndex2]
    sdY = selfieY[pointIndex1] - selfieY[pointIndex2]
    selfieRotation = np.arctan2(sdY, sdX)
    

    pdX = perfectX[pointIndex1] - perfectX[pointIndex2]
    pdY = perfectY[pointIndex1] - perfectY[pointIndex2]
    perfectRotation = np.arctan2(pdY, pdX)
    

    angle = selfieRotation-perfectRotation

    return  x, y, scalar, angle


def changePoints(selfieX, selfieY, perfectX, perfectY):
    x,y,scalar,angle = findFactors(selfieX, selfieY, perfectX, perfectY)

    for i in range(len(selfieX)):
        selfieX[i], selfieY[i] = movePoint(selfieX[i], selfieY[i], x, y)
        
    cX, cY = selfieX[i1], selfieY[i1]
    for i in range(len(selfieX)):
        selfieX[i], selfieY[i] = scalePoint(selfieX[i], selfieY[i], cX, cY, scalar)
    
    for i in range(len(selfieX)):
       selfieX[i], selfieY[i] = rotatePoint(selfieX[i], selfieY[i], cX, cY, -angle)

def giveScore(selfieX, selfieY, perfectX, perfectY):
    score = 0
    for i in range(len(selfieX)):
        score += dist(selfieX[i], selfieY[i], perfectX[i], perfectY[i])**2
    
    score = (1/np.sqrt(score))*100000
    return score


perfectX, perfectY = getPerfs()

changePoints(selfieX, selfieY, perfectX, perfectY)

print()
print(f"Your beauty score is {round(giveScore(selfieX, selfieY, perfectX, perfectY),1)}!")
for i in range(round(giveScore(selfieX, selfieY, perfectX, perfectY))):
    print(i)
    time.sleep(0.01)
#plt.plot(perfectX, perfectY)
#plt.plot(selfieX, selfieY)


#plt.plot(selfieX, selfieY)
#plt.xlim(200,800)
#plt.ylim(200,800)
#plt.gca().invert_yaxis()

#plt.show()


cv2.waitKey(delay=0)

cv2.destroyAllWindows()