import cv2
import numpy as np
import pandas as pd
import os
import feature
import cluster

ROOT_PATH = "D:\\Nhat\\term_8\\MDS\\src\\Mix"
NUM_CLUSTER = 11

X = []

list_image = []
list_image_path = []
for dirname, _, filenames in os.walk("D:\\Nhat\\term_8\\MDS\\src\\Mix"):
    for filename in filenames:
        image_path = os.path.join(dirname, filename)
        image = cv2.imread(image_path)
        list_image_path.append(image_path)
        list_image.append(image)
        avg_HSV = feature.average_HSV(image)
        print(avg_HSV)
        X.append(avg_HSV)



def clustering():

    props_kmeans_1 = cluster.kmeans_1(X, list_image_path, NUM_CLUSTER)
    (labels, center_values) = props_kmeans_1

    output_dir = "D:\\Nhat\\term_8\\MDS\\src\\kmeans2"

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
            #mod = (len(os.listdir(folder_path)) - 1) // 13
            mod = (len(os.listdir(folder_path))) / 15
            if mod > round(mod):
                mod = round(mod) + 1
            else:
                mod = round(mod)
            if mod > 1:
                cluster.kmeans_2(folder_path, mod)


def metadata():

    for dirname, child_folders, filenames in os.walk("kmeans2"):

        if len(child_folders) == 0:
            list_feature_image = []
            # Ghi metadata
            file_name = os.path.join(dirname, "metadata.txt")
            list_image_feature = feature.get_feature_bow(dirname)
            if not os.path.exists(file_name):  # Sửa từ 'exits' thành 'exists'
                # Ghi metadata vào file nếu file không tồn tại
                with open(file_name, "w") as file:
                    for key, value in list_image_feature.items():
                        file.write(f"{key}: {value}\n")

clustering()
# metadata()
