import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

#imgCrop
PLATE_WIDTH_PADDING = 1.3
PLATE_HEIGHT_PADDING = 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

MAX_DIAG_MULTIPLYER = 5
MAX_ANGLE_DIFF = 12.0
MAX_AREA_DIFF = 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3

# image Info
img = cv2.imread('333.jpg')
height, width, channel = img.shape

#Final image
longest_idx, longest_text = -1, 0
plate_imgs = []
plate_infos = []
plate_chars = []

def initial():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백으로이미지 변환
    plt.figure(figsize=(14, 8))  # 그림 크기 지정

    img_blurred = gaussian_blur(gray)  # 윤곽선 검출을 위한 블러 효과 적용, 흑백 이미지를 인자로

    return img_blurred


def gaussian_blur(gray):
    # structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # 엣지 검출을 위한 커널 행렬 설정 (모양, 크기)
    # imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    # imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)
    # #모폴로지 연산, (원본배열, 연산방법, 커널) 경계선을 검출하기 위한 사전 작업
    # # tophat = 원본 - 열림(침식 + 팽창) : 밝은 영역 강조
    # # blackhat = 닫힘(팽창 + 침식) - 원본 : 어두운 부분 강조
    #
    # imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    # gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    # #tophat, blackhat 적용

    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    #가우시안 블러 적용, 윤곽선을 더 잘 잡을 수 있도록 한다.

    return img_blurred

def median_blur(gray):
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 엣지 검출을 위한 커널 행렬 설정 (모양, 크기)
    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)
    #모폴로지 연산, (원본배열, 연산방법, 커널) 경계선을 검출하기 위한 사전 작업
    # tophat = 원본 - 열림(침식 + 팽창) : 밝은 영역 강조
    # blackhat = 닫힘(팽창 + 침식) - 원본 : 어두운 부분 강조

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    #tophat, blackhat 적용

    img_blurred = cv2.medianBlur(gray, ksize=(5))

    return img_blurred

def threshold(img_blurred):
    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=15, #픽셀 사이즈
        C=9 # 보정 상수
    )
    # Threshold 문턱값, adaptiveThreshold는 광원에도 효과적으로 엣지 추출

    contours, _ = cv2.findContours(img_thresh,
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    # 윤곽선 검출은 하얀색을 검출, findContours(이진화 이미지, 검색 방법, 근사화 방법)
    # RETR_TREE= 모든 윤곽선,계층구조 형성, CHAIN_APPROX_SIMPLE= 윤곽점들 단순화 수평, 수직 및 대각선 압축하여 끝점만 남김

    temp_result = np.zeros((height, width, channel), dtype=np.uint8) # 0으로 초기화

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1,color=(255, 255, 255))
    #이미지, 윤곽선, 윤곽선인덱스, 윤곽선,-1 : 배열 모두를 그린다.


    return contours, img_thresh, temp_result

def canny(img_blurred):
    img_canny = cv2.Canny(img_blurred, 200, 300)

    contours, _ = cv2.findContours(img_canny,
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    # 윤곽선 검출은 하얀색을 검출, findContours(이진화 이미지, 검색 방법, 근사화 방법)
    # RETR_TREE= 모든 윤곽선,계층구조 형성, CHAIN_APPROX_SIMPLE= 윤곽점들 단순화 수평, 수직 및 대각선 압축하여 끝점만 남김

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # 0으로 초기화

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    # 이미지, 윤곽선, 윤곽선인덱스, 윤곽선,-1 : 배열 모두를 그린다.

    return contours, img_canny, temp_result

def sobel(img_blurred):
    img_sobel = cv2.Sobel(img_blurred, cv2.CV_8U, 1, 0, 3)

    contours, _ = cv2.findContours(img_sobel,
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_SIMPLE)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # 0으로 초기화

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

    return contours, img_sobel, temp_result


def findContours(contours):
    temp_result = np.zeros((height, width, channel), dtype=np.uint8) # 0으로 재초기화

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h),
                      color=(255, 255, 255), thickness=2)

        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })
    # 윤곽선 이미지를 네모모양으로 표시

    return contours_dict, temp_result

#findLicensePlate
MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MAX_WIDTH, MAX_HEIGHT = 20, 80
MIN_RATIO, MAX_RATIO = 0.25, 1.0

MIN_RECT_GAP = 150
MIN_GRADIENT = 0.25

MIN_PADDING_X = 10
MAX_PADDING_X = 150
PADDING_Y = 30
def findLicensePlate(contours_dict,img_blur):
    find_contours = []

    cnt = 0
    for i in contours_dict:
        area = i['w'] * i['h']

        if area > MIN_AREA and MAX_WIDTH > i['w'] > MIN_WIDTH and MAX_HEIGHT > i['h'] > MIN_HEIGHT:
            ratio = i['w'] / i['h']
            if MAX_RATIO > ratio > MIN_RATIO:
                i['cnt'] = cnt
                cnt+=1
                find_contours.append(i)


    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    for d in find_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                      color=(255, 255, 255), thickness=2)


    for i in range(len(find_contours)):
        for j in range(len(find_contours)-(i+1)):
            if find_contours[j]['x'] > find_contours[j+1]['x']:
                temp = find_contours[j]['x']
                find_contours[j]['x'] = find_contours[j+1]['x']
                find_contours[j+1]['x'] = temp

    for i in find_contours:
        print(i['x'])

    max_cnt = 0
    max_i = 0

    for i in range(len(find_contours)):
        cnt = 0
        for j in range(i+1, len(find_contours)):
            d_x = abs(find_contours[i]['x'] - find_contours[j]['x'])
            if d_x > MIN_RECT_GAP:
                break

            d_y = abs(find_contours[i]['y'] - find_contours[j]['y'])
            if d_x == 0 or d_y == 0:
                continue
            gradient = float(d_y) / float(d_x)
            if gradient < MIN_GRADIENT:
                cnt += 1
        if cnt > max_cnt:
            max_cnt = cnt
            max_i = i

    print(max_cnt, max_i)

    # number_plate = img_blur[find_contours[max_i]['y'] - 10:find_contours[max_i]['h'] + find_contours[max_i]['y'] + 20,
    #                find_contours[max_i]['x'] - 10:140 + find_contours[max_i]['x']]
    number_plate = img_blur[find_contours[max_i]['y'] - PADDING_Y:find_contours[max_i]['h'] + find_contours[max_i]['y'] + PADDING_Y,
                   find_contours[max_i]['x'] - MIN_PADDING_X:MAX_PADDING_X + find_contours[max_i]['x']]

    resize_plate = cv2.resize(number_plate, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC + cv2.INTER_LINEAR)
    # plate_gray = cv2.cvtColor(resize_plate, cv2.COLOR_BGR2GRAY)
    _, th_plate = cv2.threshold(resize_plate, 150, 255, cv2.THRESH_BINARY)

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.imshow(th_plate)

    chars = pytesseract.image_to_string(th_plate, lang='kor', config='--psm 7 --oem 0')

    result_chars = ''
    has_digit = False
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
            if c.isdigit():
                has_digit = True
            result_chars += c
    print(result_chars)



#initial image color to gray.
img_blurred = initial()


#threshold
# contours, img_thresh, temp_result = threshold(img_blurred) # AdaptiveThreshold
contours, img_thresh, temp_result = canny(img_blurred) #canny edge
# contours, img_thresh, temp_result = sobel(img_blurred) #sobel edge

# plt.figure(1)
# plt.subplot(2,2,1)
# plt.imshow(img_blurred, cmap='gray')
plt.figure(1)
plt.subplot(2,2,2)
plt.imshow(temp_result)

#findContours
contours_dict, temp_result = findContours(contours)

plt.figure(1)
plt.subplot(2,2,3)
plt.imshow(temp_result)

findLicensePlate(contours_dict,img_blurred)


plt.show()