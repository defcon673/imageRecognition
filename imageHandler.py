import cv2
import numpy
import pickle

datafile = 'sources/dump'
height_original = 1
height_global = 1
height_delta = 1
width_original = 1
width_global = 1
width_delta = 1
windowsize_r = 32
windowsize_c = 32
cutoff_value = 256 * 0.5

def roundTo32(num):
    result = (num // 32) * 32 + 32
    delta = result - num
    return result, delta

def Split(img):
    imageList = []
    for r in range(0, img.shape[0] - windowsize_r, windowsize_r):
        for c in range(0, img.shape[1] - windowsize_c, windowsize_c):
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
        height_global = height_original
        height_delta = 0
    if(width_original % 32 != 0) :
        width_global, width_delta = roundTo32(width_original)
    else:
        width_global = width_original
        width_delta = 0

    f = open(datafile, 'wb')
    pickle.dump(filepath, f)
    pickle.dump(width_delta, f)
    pickle.dump(width_original, f)
    pickle.dump(width_global, f)
    pickle.dump(height_delta, f)
    pickle.dump(height_original, f)
    pickle.dump(height_global, f)

    res = cv2.resize(img,
                     (width_global,
                      height_global),
                     interpolation=cv2.INTER_CUBIC)
    result = Split(res)

    return result


def makeBrightValues(prediction):
    brights = [(x * 256) for x in prediction]
    return brights


def makeResultImage(prediction):
    f = open(datafile, 'rb')
    filepath = pickle.load(f)
    width_delta = pickle.load(f)
    width_original = pickle.load(f)
    width_global = pickle.load(f)
    height_delta = pickle.load(f)
    height_original = pickle.load(f)
    height_global = pickle.load(f)

    brights = makeBrightValues(prediction)
    blank_image = numpy.zeros((height_global, width_global, 3), numpy.uint8)

    max_value = 0
    for value in brights:
        if value > max_value:
            max_value = value

    x = 16
    y = 16
    for value in brights:
        cv2.circle(blank_image, (int(x), int(y)), 22,
                   ((round(255 / max_value * value),
                     round(255 / max_value * value),
                     round(255 / max_value * value)) if value > cutoff_value
                    else (0, 0, 0)),
                   cv2.FILLED)
        x += 32
        if x >= blank_image.shape[1] - 32:
            x = 0
            y += 32

    im_gray = cv2.cvtColor(blank_image, cv2.COLOR_RGB2GRAY)
    heatmap_blur = cv2.blur(im_gray, (48, 48))
    heatmap_color = cv2.applyColorMap(heatmap_blur, cv2.COLORMAP_JET)

    cv2.imwrite('heatmap.png', heatmap_color)

    img = cv2.imread(filepath, 1)
    resized_res = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
    res = cv2.addWeighted(img, 0.7, resized_res, 0.3, 0)
    cv2.imwrite("result.png", res)




