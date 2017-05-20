from network import *
from imageHandler import *
import cv2

#load image for recognition and make dataset
filepath = "image/image1.bmp"
dataset = makeDataset(filepath)

#recognize image
#dataset = numpy.load("sources/dataset.npy")
prediction = recognizeImage(dataset)

print(prediction)

#create "heatmap"