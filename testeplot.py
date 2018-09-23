import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

LANDMARK_DIR = '/Users/lucas/My Repo/Active Shape Model/muct-master/muct-landmarks/split/i000qa-fn.txt'
IMAGE_DIR = '/Users/lucas/My Repo/Active Shape Model/muct-master/a/debug/i000qa-fn.jpg'
LANDMARK_AMOUNT = 76

data = np.loadtxt(LANDMARK_DIR)
x = data[::2, ]
y = data[1::2, ]

lands = np.hstack((x, y))
_lands = np.array_split(lands, 2)
lands1 = np.ndarray.flatten(_lands[0][:])
lands2 = np.ndarray.flatten(_lands[1][:])

#imageFile = r'i000qa-fn.jpg'
#image = cv2.cvtColor(cv2.imread(IMAGE_DIR), cv2.COLOR_BGR2GRAY)

im = plt.imread(IMAGE_DIR)
plt.imshow(im)
plt.scatter(lands1, lands2)
plt.show()

# plt.figure(1)
# plt.imshow(image)
#plt.scatter(lands1, lands2)
# plt.show()

""" for i in range(len(lands1)):

    cx = lands1[1]
    cy = lands2[i]

    cv2.circle(image, (int(cx), int(cy)), 10, (255, 255, 255), -11)
    cv2.circle(image, (int(cx), int(cy)), 11, (0, 0, 255), 1)  # draw circle
    cv2.ellipse(image, (int(cx), int(cy)), (10, 10), 0, 0, 90, (0, 0, 255), -1)
    cv2.ellipse(image, (int(cx), int(cy)), (10, 10),
                0, 180, 270, (0, 0, 255), -1)
    cv2.circle(image, (int(cx), int(cy)), 1, (0, 255, 0), 1)  # draw center
    cv2.putText(image, str(i), (int(cx)+10, int(cy)-10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 180, 180)) """

#print 'plot points completed'
#cv2.imshow('ImageWindow', image)
# cv2.waitKey()

#outputfile = 'PlotPoint'+imageFile
#cv2.imwrite(outputfile, image)
