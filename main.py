from network import *
from imageHandler import *
import time
import cv2

start = time.time()
#load image for recognition and make dataset
filepath = "image/image3.bmp"
print("Load image for recognizing " + filepath)
dataset = makeDataset(filepath)

#recognize image
print("Start recognizing at " + str(time.time() - start))
prediction = recognizeImage(dataset)
print("Recognizing ended at " + str(time.time() - start))

rounded = [round(x) for x in prediction]
print(rounded)

#create "heatmap"
makeResultImage(prediction)

stop = time.time()
#print(prediction)
difference = stop - start
print("Total execution time: " + str(difference))