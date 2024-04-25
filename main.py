import cv2
import numpy as np
import pandas as pd
import os
import feature
import cluster


ROOT_PATH = 'D:\PTIT\CSDLDPT\Multimedia-Database-System\Mix'
NUM_CLUSTER = 11

def main():

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
    
    # Thực hiện kmeans lần thứ nhất
    cluster.kmeans_1(X, list_image_path, NUM_CLUSTER)

    # Thực hiện kmeans lần thứ hai
    folder = 'D:\PTIT\CSDLDPT\Multimedia-Database-System\kmeans1'
    if os.path.exists(folder):
        for i in range(NUM_CLUSTER):
            folder_path = os.path.join(folder, str(i))
            mod = len(os.listdir(folder_path)) // 13
            if mod > 1:
                cluster.kmeans_2(folder_path, mod)
        

if __name__ == '__main__': 
    main()     