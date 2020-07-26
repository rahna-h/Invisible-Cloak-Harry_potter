import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# o is default for webcam
# capture = cv2.VideoCapture("Desktop/ABC.mp4") can be done to use prerecorded videos as well!

while cap.isOpened():
    ret, back = cap.read() #Reading from the webcam
    if ret:
        cv2.imshow("image", back)
        if cv2.waitKey(5) == ord('s'):
            # save the image
            cv2.imwrite('image.jpg', back)
            break


cap = cv2.VideoCapture(0)
back = cv2.imread('./image.jpg')

while cap.isOpened():
    # take each frame
    ret, frame = cap.read()

    if ret:
        # how do we convert rgb to hsv?
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # how to get hsv value?
        # lower: hue - 10, 100, 100, higher: h+10, 255, 255
        red = np.uint8([[[0,0,255]]]) # bgr value of red
        hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
        # get hsv value of red from bgr
        # print(hsv_red)

        # threshold the hsv value to get only red colors
        l_red = np.array([170,120,70])
        u_red = np.array([180, 255, 255])
        # range increased so as to get more efficient output
        mask = cv2.inRange(hsv, l_red, u_red)

        #  part 1 is all things red
        part1 = cv2.bitwise_and(back, back, mask=mask)

        mask = cv2.bitwise_not(mask)

        # part 2 is all things not red
        part2 = cv2.bitwise_and(frame, frame, mask=mask)
        cloak = part1 + part2
        cv2.imshow("cloak", cloak)

        if cv2.waitKey(5) == ord('s'):
            break

cap.release()
cv2.destroyAllWindows()
