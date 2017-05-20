from network import *

dataset = numpy.load("sources/dataset.npy")
prediction = recognizeImage(dataset)
print(prediction)