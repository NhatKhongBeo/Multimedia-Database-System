import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle
import os


def average_BGR(image):
    avg_blue = image[:, :, 0].mean()
    avg_green = image[:, :, 1].mean()
    avg_red = image[:, :, 2].mean()

    return [avg_blue, avg_green, avg_red]


def average_HSV_simple(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_hue = image[:, :, 0].mean()
    avg_saturation = image[:, :, 1].mean()
    avg_value = image[:, :, 2].mean()

    return avg_hue, avg_saturation, avg_value

def average_HSV(image):
    # Chuyển đổi ảnh từ BGR sang HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Lọc ra những điểm ảnh không phải màu đen
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(image_hsv, lower_white, upper_white)
    
    image_hsv_filtered = cv2.bitwise_and(image_hsv, image_hsv, mask=cv2.bitwise_not(mask))

    # Tính trung bình của kênh H, S, V
    avg_hue = image_hsv_filtered[:, :, 0][image_hsv_filtered[:, :, 0] > 0].mean()
    avg_saturation = image_hsv_filtered[:, :, 1][image_hsv_filtered[:, :, 1] > 0].mean()
    avg_value = image_hsv_filtered[:, :, 2][image_hsv_filtered[:, :, 2] > 0].mean()
    
    return avg_hue, avg_saturation, avg_value

def convert_to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def SIFT(image):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    return descriptors


def kmeans_bow(all_descriptors, num_clusters):

    bow_dict = []
    kmeans = KMeans(n_clusters=num_clusters, verbose=1).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_

    return bow_dict


def create_features_bow(single_image_descriptors, BoW, num_clusters):
    feature = np.array([0] * num_clusters)
    if single_image_descriptors is not None:
        distance = cdist(single_image_descriptors, BoW, metric="euclidean")
        argmin = np.argmin(distance, axis=1)
        for j in argmin:
            feature[j] += 1
    return feature


def get_feature_bow(path):
    image_features = {}
    features = []
    # list_avg_HSV = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(dirname, filename)
                image = cv2.imread(image_path)
                avg_HSV = average_HSV(image)
                gray = convert_to_gray(image)
                sift = SIFT(gray)
                image_features[image_path] = [sift, avg_HSV]
                # image_features[avg_HSV] = [sift,avg_HSV,image_path]
                for i in sift:
                    features.append(i)

    # file_path = os.path.join(os.getcwd(), "bow_dictionary.pkl")
    num_clusters = 30
    # file_path = os.path.join(os.getcwd(), 'bow_dictionary.pkl')
    file_path = os.path.join(path, 'bow_dictionary.pkl')
    if not os.path.isfile(file_path):
        BoW = kmeans_bow(features, num_clusters)
        with open(file_path, "wb") as file:
            pickle.dump(BoW, file)
    else:
        with open(file_path, "rb") as file:
            BoW = pickle.load(file)

    for key, value in image_features.items():
        image_features[key] = {
            "Bow": create_features_bow(value[0], BoW, num_clusters).tolist(),
            "avg_Hue": value[1][0],
            "avg_Saturation": value[1][1],
            "avg_Value": value[1][2],
        }

    return image_features


def get_image_feature(image, folder_path):
    gray = convert_to_gray(image)
    sift = SIFT(gray)

    file_path = os.path.join(folder_path, "bow_dictionary.pkl")

    with open(file_path, "rb") as file:
        BoW = pickle.load(file)
    
    feature = create_features_bow(sift, BoW, 30)
    
    return feature

    






