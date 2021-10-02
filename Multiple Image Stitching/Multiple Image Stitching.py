import cv2
import glob
import matplotlib.pyplot as plt
import imutils
import numpy as np
from time import time

def display(image,caption = ''):
    plt.figure(figsize = (10,20))
    plt.title(caption)
    plt.imshow(image)
    plt.show()

def set_image(stitched):

  stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,cv2.BORDER_CONSTANT, (0, 0, 0))
  gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  c = max(cnts, key=cv2.contourArea)
  mask = np.zeros(thresh.shape, dtype="uint8")
  (x, y, w, h) = cv2.boundingRect(c)
  cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
  minRect = mask.copy()
  sub = mask.copy()

  while cv2.countNonZero(sub) > 0:
    # erode the minimum rectangular mask and then subtract
    # the thresholded image from the minimum rectangular mask
    # so we can count if there are any non-zero pixels left
    minRect = cv2.erode(minRect, None)
    sub = cv2.subtract(minRect, thresh)
  cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  c = max(cnts, key=cv2.contourArea)
  (x, y, w, h) = cv2.boundingRect(c)
  stitched = stitched[y:y + h, x:x + w]

  return stitched

def crop_image(image):
    gray = cv2.cvtColor(stitched1, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

    # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)

    # crop the image to the bbox coordinates
    result = image[y:y + h, x:x + w]
    return result
    
if __name__ == '__main__':
    
    path = '/content/drive/MyDrive/Sample/*.jpg'
    imagePaths_complete = glob.glob(path)
    imagePaths_complete = np.sort(imagePaths_complete)
    imagePaths_complete = imagePaths_complete[1:15]
    imagePaths = imagePaths_complete.copy()
    stitcher = cv2.Stitcher_create(1)

    images = []
    counter = 0
    for image in imagePaths:
      print(image)
      img = cv2.imread(image)
      images.append(img)
      counter = counter+1
 
    print(len(images))
    start = time()
    (status,stitched1) = stitcher.stitch(images)
    end = time()
    print('Time taken = ',abs(start-end))
    display(stitched1)
    #final_result = set_image(stitched1)    #Only use for less number of images of there is no distortion in the images
    final_result = crop_image(stitched1)
    
    display(final_result)
    cv2.imwrite('Stitched.jpg',final_result)
