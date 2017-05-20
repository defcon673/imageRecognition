from network import *
from imageHandler import *
import time
import cv2

start = time.time()
#load image for recognition and make dataset
filepath = "image/image1.bmp"
dataset = makeDataset(filepath)

#recognize image
#dataset = numpy.load("sources/dataset.npy")
prediction = recognizeImage(dataset)

rounded = [round(x) for x in prediction]
print(rounded)

makeResultImage(prediction)

stop = time.time()
#print(prediction)
difference = stop - start
print ("execution time: " + str(difference))
#create "heatmap"