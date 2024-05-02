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


ROOT_PATH = "D:\PTIT\CSDLDPT\Multimedia-Database-System\Mix"
NUM_CLUSTER = 11

# X = []
# list_avg_hue=[]
# list_avg_saturation=[]
# list_avg_value=[]
list_image = []
list_image_path = []
for dirname, _, filenames in os.walk("D:\PTIT\CSDLDPT\Multimedia-Database-System\Mix"):
    for filename in filenames:
        image_path = os.path.join(dirname, filename)
        list_image_path.append(image_path)
        image = cv2.imread(image_path)
        list_image.append(image)
        avg_HSV = feature.average_HSV(image)
        # list_avg_hue.append(avg_HSV[0])
        # list_avg_saturation.append(avg_HSV[1])
        # list_avg_value.append(avg_HSV[2])
        # X.append(avg_HSV)


def cluster():

    props_kmeans_1 = cluster.kmeans_1(X, list_image_path, NUM_CLUSTER)
    (labels, center_values) = props_kmeans_1

    output_dir = "D:\PTIT\CSDLDPT\Multimedia-Database-System\kmeans1"

    # Tạo đường dẫn đến file
    output_file = os.path.join(output_dir, "cluster_labels.txt")

    # Mở file để ghi dữ liệu
    with open(output_file, "w") as file:
        # Ghi nhãn của từng điểm dữ liệu
        np.savetxt(file, labels, fmt="%d")

    output_file = os.path.join(output_dir, "cluster_center.txt")

    with open(output_file, "w") as file:
        # Ghi trung tâm của các cụm

        np.savetxt(file, center_values)

    # Thực hiện kmeans lần thứ hai
    if os.path.exists(output_dir):
        for i in range(NUM_CLUSTER):
            folder_path = os.path.join(output_dir, str(i))
            mod = (len(os.listdir(folder_path)) - 1) // 13
            if mod > 1:
                cluster.kmeans_2(folder_path, mod)


def metadata():

    for dirname, child_folders, filenames in os.walk("kmeans1"):
        # sorted_child_folders = sorted(child_folders, key=lambda x: int(x))
        # for folder in sorted_child_folders:
        #     path = os.path.join(dirname, folder)
        #     dict_bow.append(feature.get_feature_bow(path))
        if len(child_folders) == 0:
            dict_bow.append(feature.get_feature_bow(dirname))
            # Ghi metadata

    # for i, item in enumerate(X):
    #     (avg_Hue, avg_Saturation, avg_Value) = item
    #     metadata[tuple(item)] = {
    #         "avg_Hue": avg_Hue,
    #         "avg_Saturation": avg_Saturation,
    #         "avg_Value": avg_Value,
    #         "cluster": labels[i],
    #     }

    images = collection.find({}, {"_id": 1, "path": 1})
    images_list = list(images)

    for i, (key, value) in enumerate(metadata.items()):
        image = images_list[i]
        metadata[key]["imageId"] = str(image["_id"])

    for item in dict_bow:
        for key_item, value_item in item.items():
            for key_meta, value_meta in metadata.items():
                if key_item == key_meta:
                    metadata[key_meta].update({"Bow": item[key_item]["Bow"]})

    for value in metadata.values():
        cluster_folder = str(value["cluster"])
        folder_path = os.path.join(
            "D:\PTIT\CSDLDPT\Multimedia-Database-System\kmeans1", cluster_folder
        )
        file_name = f"metadata_{cluster_folder}.txt"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "a") as file:
            file.write(str(value) + "\n")

        # get Bow
    dict_bow = []
