import numpy as np
import os
import cv2 as cv
import glob
import sift
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

###################################### STEP 1 & 2 ######################################

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

directory = dir_path +  "\COMP338_Assignment1_Dataset"

dataset_descriptors=[]
dataset_keypoints=[]

print("GRABBING KEYPOINTS AND DESCRIPTORS OF TRAINING IMAGES")
for root,dirs,files in os.walk(directory):
    for filename in files:
        if ".jpg" in filename:
            print(os.path.join(root, filename))
            image = cv.imread(os.path.join(root, filename),0)
            image=image.astype('float32')
            
            base_image=sift.digital_gaussian_scale_space(image)
            num_octaves=sift.compute_number_octaves(base_image.shape)
            gaussian_kernel=sift.generate_gaussian_kernel(1.6, 3)
            gaussian_images=sift.generate_Gaussian_Pyramid(base_image, num_octaves, gaussian_kernel,3)
            dog_images=sift.generate_DoG_pyramid(gaussian_images)
            keypoints=sift.findScaleSpaceExtrema(gaussian_images,dog_images,0.04,1)
            keypoints2=sift.removeDuplicateKeypoints(keypoints)
            keypoints3=sift.keypoint_to_image_size(keypoints2)
            descriptors=sift.generateDescriptors(keypoints, gaussian_images)
            if len(dataset_descriptors)==0:
                dataset_descriptors=descriptors
                dataset_keypoints=keypoints
            else:
                np.concatenate((dataset_descriptors,descriptors),axis=0)
                np.concatenate((dataset_keypoints,keypoints),axis=0)


print("TRAINING KMEANS, MAY TAKE A WHILE")
print(dataset_descriptors)
km = KMeans(init="random",n_clusters=25,n_init=10,max_iter=300)
#kmeans = KMeans(init="random",n_clusters=500,n_init=10,max_iter=300)
km.fit(dataset_descriptors)
codeword=list(range(500))
codeword_descriptor=km.cluster_centers_
codeword_dictionary=dict(zip(codeword,codeword_descriptor))
print(codeword_dictionary)


###################################### STEP 3 ######################################

def createHistograms(km):
    testHistograms = []
    testDirectory = dir_path +  "\COMP338_Assignment1_Dataset\Test"
    for root,dirs,files in os.walk(testDirectory):
        for filename in files:
            if ".jpg" in filename:
                #Grabs an image within the folder
                print(os.path.join(root, filename))
                image = cv.imread(os.path.join(root, filename),0)
                image=image.astype('float32')

                #Gets keypoints and descriptors
                base_image=sift.digital_gaussian_scale_space(image)
                num_octaves=sift.compute_number_octaves(base_image.shape)
                gaussian_kernel=sift.generate_gaussian_kernel(1.6, 3)
                gaussian_images=sift.generate_Gaussian_Pyramid(base_image, num_octaves, gaussian_kernel,3)
                dog_images=sift.generate_DoG_pyramid(gaussian_images)
                keypoints=sift.findScaleSpaceExtrema(gaussian_images,dog_images,0.04,1)
                keypoints2=sift.removeDuplicateKeypoints(keypoints)
                keypoints3=sift.keypoint_to_image_size(keypoints2)
                descript=sift.generateDescriptors(keypoints, gaussian_images)
                
                # Creates Histogram based on what the model predicts the cluster to be which is found by km.predict
                histogram, bin_edges=np.histogram(km.predict(descript), bins=500) 
                testHistograms.append(histogram)
    print(testHistograms)
    
    trainingHistograms = []
    trainingDirectory = dir_path +  "\COMP338_Assignment1_Dataset\Training"
    for root,dirs,files in os.walk(trainingDirectory):
        for filename in files:
            if ".jpg" in filename:
                #Grabs an image within the folder
                print(os.path.join(root, filename))
                image = cv.imread(os.path.join(root, filename),0)
                image=image.astype('float32')

                #Gets keypoints and descriptors
                base_image=sift.digital_gaussian_scale_space(image)
                num_octaves=sift.compute_number_octaves(base_image.shape)
                gaussian_kernel=sift.generate_gaussian_kernel(1.6, 3)
                gaussian_images=sift.generate_Gaussian_Pyramid(base_image, num_octaves, gaussian_kernel,3)
                dog_images=sift.generate_DoG_pyramid(gaussian_images)
                keypoints=sift.findScaleSpaceExtrema(gaussian_images,dog_images,0.04,1)
                keypoints2=sift.removeDuplicateKeypoints(keypoints)
                keypoints3=sift.keypoint_to_image_size(keypoints2)
                descript=sift.generateDescriptors(keypoints, gaussian_images)

                # Creates Histogram based on what the model predicts the cluster to be which is found by km.predict
                histogram, bin_edges=np.histogram(km.predict(descript), bins=500)
                trainingHistograms.append(histogram)
    print(trainingHistograms)
    
    

    return trainingHistograms, testHistograms




############################################################################################
###########################     STEP 4 & 5 BELOW  ##########################################
############################################################################################
    # note: needs a fully trained kmeans clustering model (km) to be generated before the code
    #       below can run (it's used in the histogram creation).

'''
    Classes are converted to numbers to reduce computation cost.
        airplane = 0
        cars = 1
        dog = 2
        faces = 3
        keyboard = 4
'''

'''
    Reads in all training images, and labels each image according to its class.
'''
def readTrainingImages():
    root_dir = dir_path +  "\COMP338_Assignment1_Dataset\Training"
    
    folders = glob.glob(root_dir + '*')

    training_images = []
    training_labels = []

    for folder in folders:
        for image in glob.glob(folder+'/*.jpg'):
            training_images.append(cv.imread(image))
            if 'airplane' in folder:
                training_labels.append(0)
            elif 'car' in folder:
                training_labels.append(1)
            elif 'dog' in folder:
                training_labels.append(2)
            elif 'face' in folder:
                training_labels.append(3)
            elif 'key' in folder:
                training_labels.append(4)

    return training_images, training_labels


'''
    Reads in all testing images, and labels each image according to its class to determine accuracy of predictions.
'''
def readTestingImages():
    root_dir = dir_path +  "\COMP338_Assignment1_Dataset\Test"
    folders = glob.glob(root_dir + '*')

    testing_images = []
    testing_labels = []
    for folder in folders:
        for image in glob.glob(folder+'/*.jpg'):
            testing_images.append(cv.imread(image))
            if 'airplane' in folder:
                testing_labels.append(0)
            elif 'car' in folder:
                testing_labels.append(1)
            elif 'dog' in folder:
                testing_labels.append(2)
            elif 'face' in folder:
                testing_labels.append(3)
            elif 'key' in folder:
                testing_labels.append(4)
    
    return testing_images, testing_labels


'''
    Converts all training images to normalised histograms.
'''
def getTrainingHistograms(training_labels):
    X_train = list_of_histograms[0]
    y_train = training_labels


    return X_train, y_train


'''
    Converts all testing images to normalised histograms.
'''
def getTestingHistograms(testing_labels):
    X_test = list_of_histograms[1]
    y_test = testing_labels

    return X_test, y_test


'''
    Classifies every histogram from the testing image set with a class prediction.
    Prints the confusion matrix, error rate and classification errors for each class.
'''
def getClassifications(X_train, y_train, X_test, y_test):  
    k_nn = KNeighborsClassifier(n_neighbors=3)
    k_nn.fit(X_train, y_train)
    predictions = k_nn.predict(X_test)
    accuracy = k_nn.score(X_test, y_test)
    classes = ["airplanes", "cars", "dog", "faces", "keyboard"]
    
    print("\nERROR RATE: " + str(1-accuracy))
    print(confusion_matrix(y_test, predictions))

    # Used to track which test images were correct, and which were erroneous
    # to plot a few in the evaluation.
    i = 0
    indexes_of_errors = []
    indexes_of_correct = []
    for prediction, label in zip(y_test, predictions):
        if prediction != label:
            indexes_of_errors.append(i)
        else:
            indexes_of_correct.append(i)
        i += 1


    # Splits the labels & predictions into their seperate classes to evaluate.
    y_test = np.asarray(np.array_split(y_test, 5))
    predictions = np.asarray(np.array_split(predictions, 5))
    print("\n")
    error_total = 0
    class_type = 0

    for sublist_y, sublist_p in zip(y_test, predictions):
        error_count = 0
        
        for i in range(0, len(sublist_y)):
            if sublist_p[i] != sublist_y[i]:
                error_count += 1
                
        error_total += error_count
        print("Class Label [" + classes[class_type] + "] Classification Errors: " + str(error_count))
        class_type += 1
    print("\nTotal Classification Errors: " + str(error_total))
    
    # To plot which images were right and wrong, for the final step in the evaluation for the report
    # do cv.imshow(testing_images[indexes_of_correct[0]])
    # and do cv.imshow(testing_images[indexes_of_errors[0]])


# Creates histograming for all images
list_of_histograms = createHistograms(km)

# Reads images into arrays.
training_images, training_labels = readTrainingImages()
testing_images, testing_labels = readTestingImages()

# Gets info from histograms.
X_train, y_train = getTrainingHistograms(training_labels)
X_test, y_test = getTestingHistograms(testing_labels)

# Classifies the histograms.
getClassifications(X_train, y_train, X_test, y_test)

