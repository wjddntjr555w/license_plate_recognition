import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import sys

# imgCrop
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
img = cv2.imread('scar2.png')
height, width, channel = img.shape

# Final image
longest_idx, longest_text = -1, 0
plate_imgs = []
plate_infos = []
plate_chars = []


def initial():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백으로이미지 변환
    plt.figure(figsize=(14, 8))  # 그림 크기 지정


    img_blurred = gaussian_blur(gray)  # 윤곽선 검출을 위한 블러 효과 적용, 흑백 이미지를 인자로
    # img_blurred = median_blur(gray)
    # img_blurred = gray
    plt.figure(1)
    plt.subplot(2, 2, 2)
    plt.imshow(img_blurred,cmap = 'gray')

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

    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=3)
    # 가우시안 블러 적용, 윤곽선을 더 잘 잡을 수 있도록 한다.

    return img_blurred


def median_blur(gray):
    # structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # # 엣지 검출을 위한 커널 행렬 설정 (모양, 크기)
    # imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    # imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)
    # # 모폴로지 연산, (원본배열, 연산방법, 커널) 경계선을 검출하기 위한 사전 작업
    # # tophat = 원본 - 열림(침식 + 팽창) : 밝은 영역 강조
    # # blackhat = 닫힘(팽창 + 침식) - 원본 : 어두운 부분 강조
    #
    # imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    # gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    # # tophat, blackhat 적용

    img_blurred = cv2.medianBlur(gray, ksize=(5))

    return img_blurred


def threshold(img_blurred):
    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=15,  # 픽셀 사이즈
        C=9  # 보정 상수
    )
    # Threshold 문턱값, adaptiveThreshold는 광원에도 효과적으로 엣지 추출

    contours, _ = cv2.findContours(img_thresh,
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    # 윤곽선 검출은 하얀색을 검출, findContours(이진화 이미지, 검색 방법, 근사화 방법)
    # RETR_TREE= 모든 윤곽선,계층구조 형성, CHAIN_APPROX_SIMPLE= 윤곽점들 단순화 수평, 수직 및 대각선 압축하여 끝점만 남김

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # 0으로 초기화

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    # 이미지, 윤곽선, 윤곽선인덱스, 윤곽선,-1 : 배열 모두를 그린다.

    return contours, img_thresh, temp_result


def canny(img_blurred):
    img_canny = cv2.Canny(img_blurred, 10, 100)

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
    # img_sobel = cv2.Sobel(img_blurred, cv2.CV_8U, 1, 1, 3,delta=5)

    img_sobel_x = cv2.Sobel(img_blurred,-1, 1, 0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
    img_sobel_y = cv2.Sobel(img_blurred,-1, 0, 1, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
    img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0)
    plt.figure(1)
    plt.subplot(2, 2, 4)
    plt.imshow(img_sobel, cmap='gray')
    # cv2.imshow("sobel_x", img_sobel_x)
    # cv2.imshow("sobel_y", img_sobel_y)
    # cv2.imshow("sobel", img_sobel)



    contours, _ = cv2.findContours(img_sobel,
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_SIMPLE)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # 0으로 초기화

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

    return contours, img_sobel, temp_result


def laplacian(img_blurred):
    img_laplacian = cv2.Laplacian(img_blurred, -1,ksize=13)

    contours, _ = cv2.findContours(img_laplacian,
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_SIMPLE)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # 0으로 초기화

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

    return contours, img_laplacian, temp_result

def findContours(contours):
    contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_TREE,
                                   method=cv2.CHAIN_APPROX_SIMPLE)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1,
                     color=(255, 255, 255))

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

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

    return contours_dict, temp_result


# findLicensePlate
MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MAX_WIDTH, MAX_HEIGHT = 40, 80
MIN_RATIO, MAX_RATIO = 0.25, 1.0

MIN_GRADIENT = 0.25

PADDING_X = 30
PADDING_Y = 5

def findLicensePlate_second(contours_dict, img_blur):
    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                      color=(255, 255, 255), thickness=2)

    def find_chars(contour_list):
        matched_result_idx = []

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                        and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                        and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])

            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours,
                                        unmatched_contour_idx)

            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']),
                          pt2=(d['x'] + d['w'], d['y'] + d['h']),
                          color=(255, 255, 255), thickness=2)


    return matched_result, temp_result


def findLicensePlate(contours_dict, img_blur):
    find_contours = []

    cnt = 0
    for i in contours_dict:
        area = i['w'] * i['h']

        if area > MIN_AREA and MAX_WIDTH > i['w'] > MIN_WIDTH and MAX_HEIGHT > i['h'] > MIN_HEIGHT:
            ratio = i['w'] / i['h']
            if MAX_RATIO > ratio > MIN_RATIO:
                i['cnt'] = cnt
                cnt += 1
                find_contours.append(i)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    for d in find_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                      color=(255, 255, 255), thickness=2)

    # plt.figure(1)
    # plt.subplot(2, 2, 4)
    # plt.imshow(temp_result)

    for i in range(len(find_contours)):
        for j in range(len(find_contours) - (i + 1)):
            if find_contours[j]['x'] > find_contours[j + 1]['x']:
                temp = find_contours[j]['x']
                find_contours[j]['x'] = find_contours[j + 1]['x']
                find_contours[j + 1]['x'] = temp
    cnt = 0
    for i in find_contours:
        print(cnt, i['x'], i['y'], i['w'], i['h'])
        cnt += 1

    max_cnt = 0
    before_i = 0
    max_i = 0
    for i in range(len(find_contours)):
        cnt = 0
        up_down = 0
        MIN_RECT_GAP = find_contours[i]['w'] * 15
        MIN_Y_GAP = find_contours[i]['h'] * 2
        for j in range(i + 1, len(find_contours)):
            d_x = abs(find_contours[i]['x'] - find_contours[j]['x'])
            if d_x > MIN_RECT_GAP:
                break

            d_y = abs(find_contours[i]['y'] - find_contours[j]['y'])
            if d_y > MIN_Y_GAP:
                break
            if d_x == 0 or d_y == 0:
                continue

            gradient = float(d_y) / float(d_x)
            if gradient < MIN_GRADIENT:
                if up_down == 0:
                    if find_contours[i]['y'] > find_contours[j]['y']:
                        up_down = 1  # down
                    else:
                        up_down = 2  # up

                    cnt += 1
                elif (up_down == 1 and find_contours[i]['y'] < find_contours[j]['y']) or (
                        up_down == 2 and find_contours[i]['y'] > find_contours[j]['y']):
                    break
                else:
                    cnt += 1

            if cnt > max_cnt:
                before_i = max_i
                max_cnt = cnt
                max_i = i
                max_x = d_x
                max_y = d_y
                max_rotate = up_down


    select_contours = []
    print("select_contours")
    for i in range(max_i, max_i + max_cnt):
        select_contours.append(find_contours[i])

    sel_x_max = select_contours[0]['x']
    sel_x_min = select_contours[0]['x']
    sel_y_max = select_contours[0]['y']
    sel_y_min = select_contours[0]['y']
    sel_h_max = select_contours[0]['h']
    sel_w_max = select_contours[0]['w']
    for i in range(len(select_contours)):
        if sel_x_max < select_contours[i]['x']:
            sel_x_max = select_contours[i]['x']
        if sel_x_min > select_contours[i]['x']:
            sel_x_min = select_contours[i]['x']
        if sel_y_max < select_contours[i]['y']:
            sel_y_max = select_contours[i]['y']
        if sel_y_min > select_contours[i]['y']:
            sel_y_min = select_contours[i]['y']
        if sel_h_max < select_contours[i]['h']:
            sel_h_max = select_contours[i]['h']
        if sel_w_max < select_contours[i]['w']:
            sel_w_max = select_contours[i]['w']

        print(i, select_contours[i]['x'], select_contours[i]['y'])

    print(max_cnt, max_i, find_contours[max_i]['x'], find_contours[max_i]['y'], MIN_RECT_GAP)
    print(sel_x_min, sel_x_max, sel_y_min, sel_y_max, sel_h_max, sel_w_max)

    # rows, cols = img_blur.shape[:2]
    # height_r, width_r, _ = temp_result.shape
    # if sel_y_min != select_contours[0]['y']:
    #     rotate = cv2.getRotationMatrix2D((sel_x_min, sel_y_max), (sel_y_max - sel_y_min) / 7, 1)
    #     print("DOWN")
    # else:
    #     rotate = cv2.getRotationMatrix2D((sel_x_min, sel_y_min), (sel_y_min - sel_y_max) / 7, 1)
    #     print("UP")
    # rotation_plate = cv2.warpAffine(img_blur, rotate, (cols, rows))

    number_plate = img_blur[sel_y_min - PADDING_Y:sel_y_max + PADDING_Y + sel_h_max,
                   sel_x_min - PADDING_X:sel_x_max + PADDING_X + sel_w_max]
    if number_plate != []:
        resize_plate = cv2.resize(number_plate,(0,0), fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC + cv2.INTER_LINEAR)

        # plate_gray = cv2.cvtColor(resize_plate, cv2.COLOR_BGR2GRAY)
        _, th_plate = cv2.threshold(resize_plate, 150, 255, cv2.THRESH_BINARY)

        # plt.figure(1)
        # plt.subplot(2, 2, 1)
        # plt.imshow(th_plate)
        cv2.imwrite('plate_th.jpg', th_plate)
        # cv2.imwrite('roatate_th.jpg', rotation_plate)

        chars = pytesseract.image_to_string(th_plate, lang='kor', config='--psm 7 --oem 0')

        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c
        print(result_chars)


def img_cropping(matched_result):

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
            0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    return img_cropped

# initial image color to gray.

img_blurred = initial()

# threshold
# contours, img_thresh, temp_result = threshold(img_blurred) # AdaptiveThreshold

# contours, img_thresh, temp_result = sobel(img_blurred) #sobel edge
# contours, img_thresh, temp_result = laplacian(img_blurred) #laplacian edge
# plt.figure(1)
# plt.subplot(2,2,1)
# plt.imshow(img_blurred, cmap='gray')

contours, img_thresh, temp_result = canny(img_blurred)  # canny edge

# contours, img_thresh, temp_result = laplacian(img_blurred)
plt.figure(1)
plt.subplot(2, 2, 3)
plt.imshow(temp_result, cmap='gray')

# contours, img_thresh, temp_result = sobel(img_blurred)  # sobel edge
# plt.figure(1)
# plt.subplot(2, 2, 4)
# plt.imshow(temp_result, cmap='gray')

# findContours
contours_dict, temp_result = findContours(contours)
# plt.figure(1)
# plt.subplot(2, 2, 3)
# plt.imshow(temp_result, cmap='gray')
matched_result, temp_result = findLicensePlate_second(contours_dict, img_blurred)
# plt.figure(1)
# plt.subplot(2, 2, 3)
# plt.imshow(temp_result, cmap='gray')
temp_result = img_cropping(matched_result)

plt.figure(1)
plt.subplot(2, 2, 4)
plt.imshow(temp_result, cmap='gray')

info = plate_infos[longest_idx]
img_out = img.copy()

cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(255,0,0), thickness=2)

plt.figure(1)
plt.subplot(2, 2, 1)
plt.imshow(img_out, cmap='gray')

plt.show()


print(sys.version)