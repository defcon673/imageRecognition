import cv2
import numpy

height_original = 1
height_global = 1
height_delta = 1
width_original = 1
width_global = 1
width_delta = 1
windowsize_r = 32
windowsize_c = 32

def roundTo32(num):
    result = num // 32 + 32
    delta = result - num // 32
    return result, delta

def Split(img):
    imageList = []
    for r in range(0, img.shape[0] - windowsize_r, windowsize_r):
        for c in range(0, img.shape[0] - windowsize_c, windowsize_c):
            window = img[r:r + windowsize_r, c:c + windowsize_c]
            imageList.append(window)

    result = numpy.array(imageList)
    return result

def makeDataset(filepath):
    img = cv2.imread(filepath, 1)
    height_original, width_original, channels = img.shape
    #calc new size
    if(height_original % 32 != 0) :
        height_global, height_delta = roundTo32(height_original)
    else:
        height_global, height_delta = height_original
    if(width_original % 32 != 0) :
        width_global, width_delta = roundTo32(width_original)
    else:
        width_global, width_delta = width_original

    res = cv2.resize(img, (width_global, height_global), interpolation=cv2.INTER_CUBIC)
    result = Split(res)

    return result





