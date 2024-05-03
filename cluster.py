import cv2
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.cluster import KMeans
import feature
from pymongo import MongoClient
client = MongoClient()

client = MongoClient("mongodb+srv://vphuong712:gtlp560j@cluster0.7nl7hqc.mongodb.net/")
db = client.MultimediaDB
collection = db.images



def kmeans_1(X, list_image_path, num_cluster=11):

    kmeans = KMeans(n_clusters=num_cluster,random_state =0)

    X = np.array(X)

    kmeans.fit(X)

    pred_label = kmeans.predict(X)

    output_dir = 'D:\PTIT\CSDLDPT\Multimedia-Database-System\kmeans1'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   

    dict = {}

    for i in range(len(list_image_path)):
        dict[list_image_path[i]] = pred_label[i]

    for i in range(num_cluster):
        folder_name = str(i)
        os.mkdir(os.path.join('D:\PTIT\CSDLDPT\Multimedia-Database-System\kmeans1', folder_name))

    for key, value in dict.items():
        for dirname, child_folders, filenames in os.walk('kmeans1'):
            for folder in child_folders:
                if str(value) == folder:
                    shutil.copy(key, os.path.join(dirname, folder))    

    return kmeans.labels_, kmeans.cluster_centers_



def kmeans_2(folder_path, mod):
    images = []
    list_image_path = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):        
            image_path = os.path.join(folder_path, filename)
            list_image_path.append(image_path)
            image = cv2.imread(image_path)
            avg_HSV= feature.average_HSV(image)
            images.append(avg_HSV)
    
    kmeans_second = KMeans(n_clusters=mod,random_state =0)
    kmeans_second.fit(np.array(images))
    labels = kmeans_second.predict(images)
        
    dict = {}
    for i in range((len(list_image_path))):
        dict[list_image_path[i]] = labels[i]
    

    for i in range(mod):
        folder_name =  str(i)
        os.mkdir(os.path.join(folder_path, folder_name))

    for key, value in dict.items():
        for dirname, child_folders, _ in os.walk(folder_path):
            for folder in child_folders:
                if str(value) == folder:
                    shutil.move(key, os.path.join(dirname, folder))

    # Tạo đường dẫn đến file
    output_file = os.path.join(folder_path, 'cluster_labels.txt')


    # Ghi thông tin cụm
    with open(output_file, 'w') as file:
        # Ghi nhãn của từng điểm dữ liệu
        np.savetxt(file, kmeans_second.labels_, fmt='%d')


    output_file = os.path.join(folder_path, 'cluster_center.txt')

    with open(output_file, 'w') as file:
        # Ghi trung tâm của các cụm

        np.savetxt(file, kmeans_second.cluster_centers_)

    # Bow
    dict_bow = []
    for dirname, child_folders, filenames in os.walk(folder_path):
        sorted_child_folders = sorted(child_folders, key=lambda x: int(x))
        for folder in sorted_child_folders:
            path = os.path.join(dirname, folder)
            dict_bow.append(feature.get_feature_bow(path))    

    # Ghi metadata
    metadata = {}
    for i, item in enumerate(images):
        (avg_Hue, avg_Saturation, avg_Value) = item
        metadata[tuple(item)] = {
            "avg_Hue": avg_Hue,
            "avg_Saturation": avg_Saturation,
            "avg_Value": avg_Value,
            "cluster": labels[i],
        }

    images = collection.find({}, {"_id": 1, "path": 1}) 
    images_list = list(images)

    for i, (key, value) in enumerate(metadata.items()):
        image = images_list[i]
        metadata[key]["imageId"] = str(image['_id'])

    for item in dict_bow:
        for key_item, value_item in item.items():
            for key_meta, value_meta in metadata.items():
                if key_item == key_meta:
                    metadata[key_meta].update({
                        'Bow': item[key_item]['Bow']
                    })


    for value in metadata.values():
        cluster_folder = str(value['cluster'])
        path = os.path.join(folder_path, cluster_folder)
        last_folder = os.path.split(folder_path)[-1]
        file_name = f"metadata_{last_folder}.{cluster_folder}.txt"
        file_path = os.path.join(path, file_name)
        with open(file_path, 'a') as file:
            file.write(str(value) + '\n') 

   


    
    




