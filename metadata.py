import cv2
import numpy as np
import pandas as pd
import os
import feature
import cluster
from pymongo import MongoClient

client = MongoClient()

client = MongoClient("mongodb+srv://vphuong712:gtlp560j@cluster0.7nl7hqc.mongodb.net/")
db = client.MultimediaDB
collection = db.images


ROOT_PATH = 'D:\PTIT\CSDLDPT\Multimedia-Database-System\Mix'
NUM_CLUSTER = 11

def metadata():

    # Tạo bộ dữ liệu
    X = []
    list_avg_hue=[]
    list_avg_saturation=[]
    list_avg_value=[]
    list_image = []
    list_image_path=[]
    for dirname, _, filenames in os.walk('D:\PTIT\CSDLDPT\Multimedia-Database-System\Mix'):
        for filename in filenames:
            image_path = os.path.join(dirname, filename)
            list_image_path.append(image_path)
            image = cv2.imread(image_path)
            list_image.append(image)
            avg_HSV= feature.average_HSV(image)
            list_avg_hue.append(avg_HSV[0])
            list_avg_saturation.append(avg_HSV[1])
            list_avg_value.append(avg_HSV[2])
            X.append(avg_HSV)
    
    # print(X)
    
    # Thực hiện kmeans lần thứ nhất
    cluster_1 = cluster.kmeans_1(X, list_image_path, NUM_CLUSTER)

    # Thực hiện kmeans lần thứ hai
    folder = 'D:\PTIT\CSDLDPT\Multimedia-Database-System\kmeans1'


    tmp = []

    if os.path.exists(folder):
        for i in range(NUM_CLUSTER):
            folder_path = os.path.join(folder, str(i))
            mod = len(os.listdir(folder_path)) // 13
            if mod > 1:
                tmp.append(cluster.kmeans_2(folder_path, mod))
    
    cluster_2 = {}

    for item in tmp:
        for key, value in item.items():
            cluster_2[key] = value
    
    res = {}

    for key, value in cluster_1.items():
        if key in cluster_2:
            res[key] = cluster_2[key]
        else:
            res[key] = value
    
    # print(res)
    
    metadata = {}
    for key, value in res.items():
        (avg_Hue, avg_Saturation, avg_Value) = key
        metadata[key] = {
            'avg_Hue': avg_Hue,
            'avg_Saturation': avg_Saturation,
            'avg_Value': avg_Value,
            'cluster' : value
        }
    
    images = collection.find({}, {"_id": 1, "path": 1}) 
    images_list = list(images)

    for i, (key, value) in enumerate(metadata.items()):
        image = images_list[i]
        metadata[key]['imageId'] = str(image['_id'])

    
    
metadata()