from network import *
import cv2

#load image for recognition and make dataset
im = cv2.imread("image/image1.bmp", 1)


#recognize image
dataset = numpy.load("sources/dataset.npy")
prediction = recognizeImage(dataset)
print(prediction)

#create "heatmap"