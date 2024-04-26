import cv2
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.cluster import KMeans
import feature


def kmeans_1(X, list_image_path, num_cluster=11):

    kmeans = KMeans(n_clusters=num_cluster,random_state =0)

    X = np.array(X)

    kmeans.fit(X)

    pred_label = kmeans.predict(X)


    output_dir = 'D:\PTIT\CSDLDPT\Multimedia-Database-System\kmeans1'

    # Kiểm tra xem thư mục tồn tại chưa, nếu không, tạo mới
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Tạo đường dẫn đến file
    output_file = os.path.join(output_dir, 'cluster_labels.txt')


    # Mở file để ghi dữ liệu
    with open(output_file, 'w') as file:
        # Ghi nhãn của từng điểm dữ liệu
        np.savetxt(file, kmeans.labels_, fmt='%d')


    output_file = os.path.join(output_dir, 'cluster_center.txt')

    with open(output_file, 'w') as file:
        # Ghi trung tâm của các cụm

        np.savetxt(file, kmeans.cluster_centers_)


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
    
    metadata = {}
    for i in range(len(X)):
        metadata[tuple(X[i])] = str(pred_label[i])
    
    return metadata
    

def kmeans_2(folder_path, mod):
    images = []
    list_image_path = []
    for filename in os.listdir(folder_path):
        # print(folder_path)
        image_path = os.path.join(folder_path, filename)
        # print(image_path)
        list_image_path.append(image_path)
        image = cv2.imread(image_path)
        # print(image)
        avg_HSV= feature.average_HSV(image)
        images.append(avg_HSV)
    
    kmeans_second = KMeans(n_clusters=mod,random_state =0)
    kmeans_second.fit(np.array(images))
    label = kmeans_second.predict(images)
        
    dict = {}
    for i in range((len(list_image_path))):
        dict[list_image_path[i]] = label[i]
    

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


    # Mở file để ghi dữ liệu
    with open(output_file, 'w') as file:
        # Ghi nhãn của từng điểm dữ liệu
        np.savetxt(file, kmeans_second.labels_, fmt='%d')


    output_file = os.path.join(folder_path, 'cluster_center.txt')

    with open(output_file, 'w') as file:
        # Ghi trung tâm của các cụm

        np.savetxt(file, kmeans_second.cluster_centers_)
    
    last_folder = os.path.split(folder_path)[-1]

    metadata = {}
    for i in range(len(images)):
        metadata[tuple(images[i])] = f'{last_folder}.{label[i]}'
    
    return metadata


