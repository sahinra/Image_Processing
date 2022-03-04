import numpy as np
import cv2
import os

# INPUT_PATH = "leaf_dataset"
OUTPUT_PATH = "output_masks/dataset1_masks/"
OUTPUT_PATH2 = "output_masks/dataset2_masks/"
  
# if not os . path . exists ( INPUT_PATH ) :
#     print ( " Incorrect path: " + INPUT_PATH )
#     exit (1)
if not os . path . exists ( OUTPUT_PATH ) :
    print ( " Incorrect path: " + OUTPUT_PATH )
    exit (1)

kernel = np.ones((5,5), np.uint8)
imageCount = 0
dataSet1Directory = r'leaf_dataset/leaf_dataset/leaves_testing_set_1/color_images'
dataSet2Directory = r'leaf_dataset/leaf_dataset/leaves_testing_set_2/color_images'

# read all images in the dataset directory  
for fileName in os.listdir(dataSet2Directory):
    f = os.path.join(dataSet2Directory, fileName)
    if os.path.isfile(f):
        #read the image in grayscale
        inputImage = cv2.imread(f, 0)

        #blur the image
        blurred = cv2.GaussianBlur(inputImage, (3,3), 0)

        # use threshold
        ret, thresh = cv2.threshold(inputImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # use openning morphological operation
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                            kernel, iterations = 1)

        # Show the mask 
        cv2.imshow('image', mask)
        cv2.waitKey(0)       

        # save the mask 
        if(cv2.imwrite(OUTPUT_PATH2 + "leaf" + str(imageCount) + ".png", mask)):
            imageCount = imageCount + 1

# check if all saved
if(imageCount == 150):
    print("All images succesfully saved")
    exit(0)
else:
    print("{0} images succesfully saved".format(imageCount))
    exit(1)
