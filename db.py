import os
import pymongo
from pymongo import MongoClient
client = MongoClient()

client = MongoClient("mongodb+srv://vphuong712:gtlp560j@cluster0.7nl7hqc.mongodb.net/")
database = client.MultimediaDB
collection = db.images

ROOT_PATH = 'D:\PTIT\CSDLDPT\Multimedia-Database-System\Mix'

# for filename in os.listdir(ROOT_PATH):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         path = os.path.join(ROOT_PATH, filename)
#         with open(path, 'rb') as f:
#             # Đọc dữ liệu từ tệp
#             image_data = f.read()
#             # Chèn dữ liệu vào MongoDB
#             collection.insert_one({'filename': filename,'path': path, 'image_data': image_data})
#             print(f'{filename} inserted into MongoDB')

# print('All images inserted into MongoDB')