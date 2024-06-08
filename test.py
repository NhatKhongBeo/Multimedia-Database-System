import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import feature

# Mở tệp để đọc

def euclidean_distance(vector1, vector2):
    # Tính khoảng cách Euclidean giữa hai vector
    return np.linalg.norm(vector1 - vector2)

def find_closest_vectors(target_vector, vectors):
    min_distance = float('inf')  # Khởi tạo khoảng cách nhỏ nhất là vô cùng lớn
    closest_vectors = []  # Danh sách để lưu trữ các vector gần nhất

    # Lặp qua từng vector trong danh sách và tìm khoảng cách Euclidean
    for vector in vectors:
        distance = euclidean_distance(target_vector, vector)
        
        # Nếu khoảng cách nhỏ hơn khoảng cách nhỏ nhất hiện tại, cập nhật khoảng cách và danh sách vector gần nhất
        if distance < min_distance:
            min_distance = distance
            closest_vectors = [vector]
        elif distance == min_distance:  # Nếu có nhiều vector có cùng khoảng cách nhỏ nhất, thêm chúng vào danh sách
            closest_vectors.append(vector)
    
    return min_distance, closest_vectors

# Đọc vector từ file
def read_vectors_from_file(file_path):
    vectors = []
    with open(file_path, 'r') as file:
        for line in file:
            numbers = [float(num) for num in line.split()]
            vectors.append(np.array(numbers))
    return vectors

# Thay đổi 'path_to_your_file.txt' thành đường dẫn của tệp bạn muốn sử dụng
file_path = "MDS/kmeans1/cluster_center.txt"  # Thay đổi đường dẫn đến tệp của bạn
# Đọc các vector từ tệp
vectors_in_file = read_vectors_from_file(file_path)

# Thay đổi 'target_vector' thành vector mà bạn muốn so sánh với các vector trong tệp
target_vector = np.array([1.0, 2.0, 3.0])  # Thay bằng vector của bạn

# Tìm khoảng cách gần nhất và các vector gần nhất
min_distance, closest_vectors = find_closest_vectors(target_vector, vectors_in_file)

# In ra kết quả
print("Khoảng cách gần nhất:", min_distance)
print("Các vector gần nhất:")
for vector in closest_vectors:
    print(vector)
