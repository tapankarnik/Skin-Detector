import numpy as np
import cv2
import math
import pickle
import dlib
import argparse
import imutils
from imutils import face_utils

from numba import jit

def get_arguments():
    parser = argparse.ArgumentParser(description="Skin Detection")
    parser.add_argument("--input", type=str, default="suzy.jpeg", help="Enter the filename of the image")
    parser.add_argument("--width", type=int, default=None, help="Enter the desired width of the image. Image will be resized.")
    return parser.parse_args()

def get_ellipse(image, rects):
    masks = []
    ellipses = []
    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        print("face at "+ str([x,y,w,h]))
        print("Gota face")
        im = image.copy()
        # cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        # cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),-1)
        canvas = np.zeros_like(im)
        canvas[y:y+h, x:x+w,:] = im[y:y+h, x:x+w,:]
        shape = predictor(im, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(im, (x, y), 1, (0, 0, 255), -1)
        cv2.imwrite("outputs/rect.jpg", im)

        left_eye_center = tuple(shape[38])
        right_eye_center = tuple(shape[43])
        eye_center = ((left_eye_center[0]+right_eye_center[0])//2, (left_eye_center[1]+right_eye_center[1])//2)
        eye_center = tuple(map(int, eye_center))
        distance_eyes = int(math.sqrt(pow(right_eye_center[1]-left_eye_center[1], 2) + pow(right_eye_center[0]-left_eye_center[0], 2)))
        axes = (1.6*(distance_eyes//2), 1.8*(distance_eyes//2))
        axes = tuple(map(int, axes))

        h, w, d = image.shape
        ellipse = np.zeros((h, w), np.uint8)
        cv2.ellipse(ellipse, eye_center, axes, 0, 0, 360, 255, -1)
        ellipses.append(ellipse)
        mask = cv2.bitwise_and(image, image, mask=ellipse)
        masks.append(mask)
        print("Got mask!")
    h, w, d = image.shape
    mask = np.zeros((h, w, d), np.uint8)
    ellipse = np.zeros((h, w), np.uint8)
    print("There are "+str(len(masks))+" Masks found!!")
    for i in range(len(masks)):
        mask = cv2.bitwise_or(mask,masks[i])
        ellipse = cv2.bitwise_or(ellipse,ellipses[i])
    return mask, ellipse

def get_sobel(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_mask = cv2.Canny(image, 50, 256)
    cv2.imwrite("outputs/sobel.jpg",sobel_mask)
    return sobel_mask

def get_dilation(image):
    kernel = np.ones((5,5), np.uint8)
    image = cv2.dilate(image, kernel, iterations=2)
    cv2.imwrite("outputs/dilated.jpg",image)
    return image

def flip_mask(image, dilated_image, ellipse):
    '''
    This function is for flipping the values of the mask that lie inside the face region.
    Then the mask is applied on the image and only skin region should remain.
    '''

    dilated_image = cv2.bitwise_and(image, image, mask=dilated_image)
    image = cv2.subtract(image, dilated_image)
    return image 

@jit(nopython=True)
def calc_Histogram(image, ellipse):
    hist_3D = np.zeros((256,256,256), dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(ellipse[i,j]==255):
                hist_3D[image[i,j,0], image[i,j,1], image[i,j,2]]+=1
    hist_3D[0][0][0] = 0
    hist_3D[0][128][128] = 0
    return hist_3D

def normalize_histogram(hist_3D):
    hist_3D = optimized_normalization(hist_3D)
    
    # hist_3D = hist_3D/27
    return hist_3D

@jit(nopython=True)
def optimized_normalization(hist_3D):
    new_hist_3D = np.zeros_like(hist_3D)
    for (i,j,k), _ in np.ndenumerate(hist_3D):
        if(i<253 and j<253 and k<253):
            for m in range(3):
                for n in range(3):
                    for p in range(3):
                        new_hist_3D[i+1,j+1,k+1]+=hist_3D[i+m,j+n,k+p]
    return new_hist_3D

@jit(nopython=True)
def get_indices(image, h, lower, upper):

    indices = list()
    for i in range(0,256):
        for j in range(0,256):
            for k in range(256):
                if h[i,j,k]>=lower and h[i,j,k]<=upper:
                    indices.append([i,j,k])
        
    return np.array(indices)

def calc_dynamic_thresholds(h, image, real_image, mask_image, ID):

    mean = np.mean(h[h>0])
    std = np.std(h[h>0])
    # factor = 0.3
    lower_freq = mean
    upper_freq = np.max(h)
    print("Mean "+str(mean))
    print("Std Deviation "+str(std))
    print("Max "+str(np.max(h)))
    print("Min "+str(np.min(h)))
    indices = get_indices(image, h, lower_freq, upper_freq)

    if(indices.shape[0]>16000000):
        print("Too large. skipping.")
        return mask_image
    # temp_img = get_values(indices, real_image, mask_image)
    lower = indices[0]
    upper = indices[-1]
    temp_img = cv2.inRange(real_image, lower, upper)
    cv2.imwrite("outputs/thresholded"+str(ID)+".jpg",temp_img) 
    return temp_img

# @jit(nopython=True)
def get_values(indices, image, temp_img):

    print ("number of indices: ", indices.shape)
    for index in indices:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if((image[i,j,0]==index[0]) and (image[i,j,1]==index[1]) and (image[i,j,2]==index[2])):
                    temp_img[i,j] = 1
    return temp_img

if __name__=='__main__':
    args = get_arguments()
    image_BGR = cv2.imread('input/'+args.input)

    image_BGR = imutils.resize(image_BGR, args.width)
    img = image_BGR.copy()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    #################
    detector = dlib.get_frontal_face_detector()

    #Download this file from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')  
    rects = detector(image_BGR, 1)
    #################

    #This condition is for the dynamic approach
    if rects is not None:
        mask_img, ellipse = get_ellipse(img, rects) 
        sobel_image = get_sobel(mask_img)
        dilated_image = get_dilation(sobel_image)
        smooth_image = flip_mask(mask_img, dilated_image.copy(), ellipse) #The.copy() is so that dilated_image doesn't get altered.

        rgb = smooth_image.copy()
        hsv = cv2.cvtColor(smooth_image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(smooth_image, cv2.COLOR_BGR2YCR_CB)
        cv2.imwrite("outputs/masked_hsv.jpg",hsv)
        cv2.imwrite("outputs/masked_ycrcb.jpg",ycrcb)
        print("Calculating RGB 3D Histogram")
        h1 = calc_Histogram(rgb, ellipse)
        print("Calculating HSV 3D Histogram")
        h2 = calc_Histogram(hsv, ellipse)
        print("Calculating YCrCb 3D Histogram")
        h3 = calc_Histogram(ycrcb, ellipse)


        print("Normalizing RGB 3D Histogram")
        h1 = normalize_histogram(h1)
        # print("Normalizing HSV 3D Histogram")
        # h2 = normalize_histogram(h2,2)
        # print("Normalizing YCrCb 3D Histogram")
        # h3 = normalize_histogram(h3,3)

        mask_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask_image_hsv = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask_image_ycrcb = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        masked_image_rgb = calc_dynamic_thresholds(h1, rgb, img, mask_image, 1)
        masked_image_hsv  = calc_dynamic_thresholds(h2, hsv, img_hsv, mask_image_hsv, 2)
        masked_image_ycrcb = calc_dynamic_thresholds(h3, ycrcb, img_ycrcb, mask_image_ycrcb, 3)
    #Else is for the explicit approach (thresholding)
    else:
        print("No Face Detected!")
    
    cv2.imshow("original",img)
    cv2.imshow("mask",mask_img)
    cv2.imshow("sobel_image",sobel_image)
    cv2.imshow("dilated_image",dilated_image)
    cv2.imshow("smooth_image", smooth_image)
    cv2.imshow("RGB", rgb)
    cv2.imshow("HSV", hsv)
    cv2.imshow("YCRCB", ycrcb)

    cv2.imshow("masked_skin_rgb", cv2.bitwise_and(img,img,mask=masked_image_rgb))
    cv2.imshow("masked_skin_hsv", cv2.bitwise_and(img,img,mask=masked_image_hsv))
    cv2.imshow("masked_skin_ycrcb", cv2.bitwise_and(img,img,mask=masked_image_ycrcb))
    combined = cv2.bitwise_or(cv2.bitwise_or(masked_image_rgb, masked_image_ycrcb), masked_image_ycrcb)
    cv2.imwrite("outputs/final_output.jpg", combined)
    combined = cv2.bitwise_and(img,img,mask=combined)
    cv2.imshow("combined", combined)
    cv2.imwrite("outputs/combined.jpg", combined)

    cv2.imshow("Comparison", np.hstack([img, combined]))
    cv2.waitKey(0)

    cv2.destroyAllWindows()


    