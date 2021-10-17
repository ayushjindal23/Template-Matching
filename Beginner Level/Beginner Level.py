#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Librarier
import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[3]:


# loading test image
img = cv2.imread('test_image.png')

# converting test image into grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Loding template image and converting into gray scale
template = cv2.imread('template_image.png',0)

# to get the column and rows value in the reverse order
w, h = template.shape[::-1]


# In[15]:


# Method to match template
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)

# to get the  brightest point
threshold = 0.99
loc = np.where( res >= threshold)

# to  draw the rectangles and to iterate over the matched templates
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)


# In[16]:


# Print image
cv2.imshow('res.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows

