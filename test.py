import cv2
import numpy as np
import time
import sys
# image = cv2.imread('whitelane.jpg')
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #Drawing the lines
    # if lines is None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    # print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]

    #finding out the edges
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2, # 6
                            theta=np.pi/100, # 60
                            threshold=10, # 160
                            lines=np.array([]),
                            minLineLength=30,#40
                            maxLineGap=10)#25
    image_with_lines = drow_the_lines(image, lines)
    return image_with_lines

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
print('FPS:', fps)
prev_time = 0;
FPS = 144.7
while(cap.isOpened()):
    ret, frame = cap.read()
    str = "FPS : %d" % fps # 화면에 프레임 글씨 표시
    cv2.putText(frame, str, (0, 50), cv2.FONT_ITALIC, 1.5, (255, 255, 0),2) # 화면에 프레임 글씨 표시 , #1.5(크기) 2(굵기) # 0,50(x,y)
    current_time = time.time() - prev_time # 프레임 바꾸기
    if (ret is True) and (current_time > 1./ FPS):# 프레임 바꾸기
        prev_time = time.time() # 프레임 바꾸기
        frame = process(frame)
        cv2.imshow('frame', frame)
        #break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()