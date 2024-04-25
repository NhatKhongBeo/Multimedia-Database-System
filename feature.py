import cv2
import numpy as np
import pandas as pd

def average_BGR(image):
    avg_blue = image[:,:,0].mean()
    avg_green = image[:,:,1].mean()
    avg_red = image[:,:,2].mean()

    return [avg_blue,avg_green,avg_red]

def average_HSV(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    avg_hue = image[:,:,0].mean()
    avg_saturation = image[:,:,1].mean()
    avg_value = image[:,:,2].mean()

    return avg_hue, avg_saturation, avg_value

