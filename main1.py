import cv2
import os
import numpy as np
import feature

def Vector(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    vectors = []

    for line in lines:
        
        numbers = [float(num) for num in line.split()]
        
        if len(numbers) == 3:
            vectors.append(numbers)
        else:
            print("Error Line: ", line)
    return vectors

def euclidean_distance(vector1, vector2):
    # Tính khoảng cách Euclidean giữa hai vector
    return np.linalg.norm(vector1 - vector2)

def find_vectors(target_vector, vectors):
    min_distance = float('inf')  
    closest_vectors = []  
    closest_index = []

    for i,vector in enumerate(vectors):
        distance = euclidean_distance(target_vector, vector)
        
        if distance < min_distance:
            min_distance = distance
            closest_vectors = np.array(vector)
            closest_index = np.array(i)
    
    return min_distance, closest_index, closest_vectors

def load_image():
    X = []
    image_folder = "MDS/image/2"
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename) 
            image = cv2.imread(image_path)
            avg_HSV= feature.average_HSV(image)
            X=np.array(avg_HSV)
    return X

def find_cluster(x):
    folder_path = "MDS/kmeans1"
    filename = "cluster_center.txt"
    file_path = os.path.join(folder_path, filename)
    while os.path.exists(file_path):
        v = Vector(file_path)
        min_distance, closest_index, closest_vectors = find_vectors(x, v)
        folder_path = folder_path+ "/"+ str(closest_index.item())
        file_path = os.path.join(folder_path, filename)
        # print(folder_path)
        # print(min_distance, closest_index, closest_vectors)
    return folder_path
if __name__ == '__main__':
    print(find_cluster(load_image()))