import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Load an color image in grayscale
frame = cv2.imread('sample_dog.jpg')
w, h, _ = frame.shape
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)

inp_path = 'scan'
all_inps = os.listdir(inp_path)

for img in all_inps:
    scan = cv2.imread(inp_path+'/'+img, cv2.IMREAD_UNCHANGED)
    for i in range(scan.shape[0]):
        for j in range(scan.shape[1]):
            scan[i,j,0] = scan[i,j,0] * (scan[i,j,3] / 255)
            scan[i,j,1] = scan[i,j,1] * (scan[i,j,3] / 255)
            scan[i,j,2] = scan[i,j,2] * (scan[i,j,3] / 255)
            # if scan[i,j,3] == 0:
            #     scan[i,j] = [0,0,0,0]
    cv2.imwrite('scan2/'+img, scan)
    print('saved scan2/',img)

if scan is not None:
    cv2.imshow(' ', scan)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('key:', key)
else:
    print('img is None')
