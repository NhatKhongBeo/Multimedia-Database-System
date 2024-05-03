import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
import os

def average_BGR(image):
    avg_blue = image[:,:,0].mean()
    avg_green = image[:,:,1].mean()
    avg_red = image[:,:,2].mean()

    return [avg_blue,avg_green,avg_red]

def average_HSV(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    avg_hue = image[:,:,0].mean()
    avg_saturation = image[:,:,1].mean()
    avg_value = image[:,:,2].mean()

    return avg_hue, avg_saturation, avg_value

def convert_to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def SIFT(image):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    return  descriptors

def kmeans_bow(all_descriptors, num_clusters):

    bow_dict = []
    kmeans = KMeans(n_clusters=num_clusters, verbose = 1).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_
    
    return bow_dict

def create_features_bow(single_image_descriptors, BoW, num_clusters):
    feature = np.array([0] * num_clusters)
    if single_image_descriptors is not None:
        distance = cdist(single_image_descriptors, BoW, metric='euclidean')
        argmin = np.argmin(distance, axis=1)
        for j in argmin:
            feature[j] += 1
    return feature

def get_feature_bow(path):
    image_features = {}
    features = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dirname, filename)
                image = cv2.imread(image_path)
                gray = convert_to_gray(image)
                sift = SIFT(gray)
                image_features[average_HSV(image)] = sift
                for i in sift:
                    features.append(i)                

    num_clusters = 50
    file_path = os.path.join(os.getcwd(), 'bow_dictionary.pkl')
    if not os.path.isfile(file_path):
        BoW = kmeans_bow(features, num_clusters)
        with open(file_path, 'wb') as file:
            pickle.dump(BoW, file)
    else:
        with open(file_path, 'rb') as file:
                BoW = pickle.load(file)

    for key, value in image_features.items():
        image_features[key] = {'Bow': create_features_bow(value, BoW, num_clusters).tolist()}
    
    return image_features


