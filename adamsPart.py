"""
    HOW TO USE:
        after all code, read in a photo using cv.imread
        then call histogramMaker(img, km) to return the histogram as a feature array aswell as visually see it
        i will be working on it to grab all of them but that can be done while you crack on with your part
        any questions just ask 

"""


import numpy as np
import cv2 as cv
import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


################### Step 2 but only 300 words for now (not my part so quickly and poorly done) ###################
image_list = []
all_des = []
print("GRABBING IMAGES")
for filename in glob.glob(r'C:\Users\Adam\Desktop\COMP338_Assignment1_Dataset\Training\dog\*.jpg'): #gets all images
    im = cv.imread(filename)
    image_list.append(im)

print("GETTING DESCRIPTORS")
for img in image_list: # Gets keypoints and descriptors for image (to be replaced with tylers code)
    grey= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kp, des = cv.SIFT.create().detectAndCompute(grey, None) 
    all_des.append(des)

all_desarray=np.array(all_des, dtype=np.ndarray) #changes descriptor list to array
all_des=np.concatenate(all_des, axis=0)
print('TRAINING KMEANS DOGS, WILL TAKE A FEW MINS...')
km = KMeans(n_clusters=300, random_state=0).fit(all_des)

######################################################################################

def histogramMaker(img, km):
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY) #Convert to grayscale 
    kp, descript = cv.SIFT.create().detectAndCompute(gray, None) # Gets keypoints and descriptors for image (to be replaced with tylers code)
    histogram, bin_edges=np.histogram(km.predict(descript)) #Histogram as feature vector
    plt.hist(km.predict(descript), bins = bin_edges) # Histogram visualised based on model
    plt.show()
    return histogram


