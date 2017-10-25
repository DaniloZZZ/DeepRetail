import numpy as np
import cv2 ,drdata
from drdata import DRData

def hw(data):
    i  = open('./data/img.png','rb')
    i2 = open('./data/img2.png','rb')
    img = i.read()
    img2 = i2.read()
    i2.close()
    i.close()
    print type(img)
    return img+"newdrtimg"+img2

