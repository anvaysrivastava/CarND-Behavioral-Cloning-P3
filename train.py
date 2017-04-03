import csv
import cv2
import numpy as np

lines = []

with open('data/driving_log.csv') as csv_file:
	reader = csv.reader(csv_file)
	for line in reader:
		lines.append(line)

center_images = []
measurements = []

for line in lines:
	if ".jpg" in line[0]:	
		image = cv2.imread("data/" + line[0])
		center_images.append(image)
		measurements.append(float(line[3]))
X_train = np.array(center_images)
Y_train = np.array(measurements)
print(X_train[0])
print(Y_train[0])	
