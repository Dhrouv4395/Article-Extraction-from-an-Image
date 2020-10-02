import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
import tkinter as tk
from PIL import Image, ImageTk 
import cv2

root = tk.Tk()
image = Image.open('913e88_169803_2.jpg')
#image = image.resize((250, 250), Image.ANTIALIAS)
img = ImageTk.PhotoImage(image)
imgtext = pytesseract.image_to_string(image)

#print(imgtext)
image = cv2.imread('de416a_169791_5.jpg')
#-------------------------------------------------------------------------------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert2grayscale
(thresh, binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # convert2binary

(contours, _) = cv2.findContours(~binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
# find contours
for contour in contours:
    """
    draw a rectangle around those contours on main image
    """
    [x,y,w,h] = cv2.boundingRect(contour)
    cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)

#------------------------------------------------------------------------------------------------------------------------
import numpy as np
mask = np.ones(image.shape[:2], dtype="uint8") * 255 # create blank image of same dimension of the original image
(contours, _) = cv2.findContours(~binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
heights = [cv2.boundingRect(contour)[3] for contour in contours] # collecting heights of each contour
avgheight = sum(heights)/len(heights) # average height
# finding the larger contours
# Applying Height heuristic
for c in contours:
    [x,y,w,h] = cv2.boundingRect(c)
    if h > (avgheight):
        cv2.drawContours(mask, [c], -1, 0, -1)
#--------------------------------------------------------------------------------------------------------------------------
from pythonRLSA import rlsa
import math
x, y = mask.shape
value = max(math.ceil(x/100),math.ceil(y/100))+22 #heuristic
mask = rlsa.rlsa(mask, True, False, value) #rlsa application
#--------------------------------------------------------------------------------------------------------------------------
(contours, _) = cv2.findContours(~mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # find contours
mask2 = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    if w > 0.60*image.shape[1]: # width heuristic applied
        title = image[y: y+h, x: x+w] 
        mask2[y: y+h, x: x+w] = title # copied title contour onto the blank image
        image[y: y+h, x: x+w] = 255 # nullified the title contour on original image
#--------------------------------------------------------------------------------------------------------------------------
from PIL import Image
from PIL import ImageOps
import pytesseract
title = pytesseract.image_to_string(Image.fromarray(mask2))
content = pytesseract.image_to_string(Image.fromarray(image))

print(title)
print('*'*50)
#print(content)
print(avgheight)
cv2.imshow('b',image)
cv2.waitKey(0)
#w1 = tk.Label(root, image=img).pack(side="right")

#explanation ='{TEXT OUTPUT}\n\n'+imgtext.lower()

#w2 = tk.Label(root, 
#              justify=tk.LEFT,
#              padx = 10, 
#              text=explanation,bg='dark green',fg='light green').pack(side="left")
#root.mainloop()
