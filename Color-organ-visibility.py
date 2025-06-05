#!/usr/bin/env python
# coding: utf-8

#INSTALL DEPENDENCIES IF NOT ALREADY INSTALLED
# pip install opencv-python numpy matplotlib PyWavelets

#IMPORT LIBRARIES
import cv2
import numpy as np
import matplotlib.pyplot as plt

#LOADING THE ORIGINAL .TIF IMAGE USING OPENCV
image_path = "C://David//DYNAMIC_BIO-IMAGING_LAB//MICE_Project//Mouse2.tif"     #Replace with the path of your image
color_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

#NORMALIZATION IF PIXEL VALUES EXCEED 255
if color_image.max() > 255:
    color_image = (color_image / color_image.max()) * 255  # Normalize to [0, 255]
    
# Convert to uint8 for proper visualization
color_image = color_image.astype(np.uint8)

#DISPLAY COLOR IMAGE
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)) #Convert BGR to RGB for correct display in opencv

#CROPPING (SELECTING ROI INTERACTIVELY)
roi = cv2.selectROI("Select Region", color_image, fromCenter=False, showCrosshair=True)

#EXTRACT CROP COORDINATES
x, y, w, h = roi

#CROP IMAGE
cropped_color = color_image[y:y+h, x:x+w]

#CLOSE SELECTION WINDOW
cv2.destroyAllWindows()

#DISPLAY COLOR IMAGE
plt.imshow(cv2.cvtColor(cropped_color,cv2.COLOR_BGR2RGB))
plt.axis("off")

#PROCESSING COLORED IMAGE AS LABSPACE COLOR SPACE
lab_color = cv2.cvtColor(cropped_color, cv2.COLOR_BGR2LAB)
#SPLIT LAB CHANNELS
l_channel, a_channel, b_channel = cv2.split(lab_color)

#ANALYZING PIXEL DISTRIBUTION (Plotting histogram of pixel intensities)
plt.figure(figsize=(8,6))
plt.hist(l_channel.ravel(), bins=256, range=[0,255], color='black', alpha=0.7)
plt.title("Pixel Intensity Distribution")
plt.xlabel("Intensity Value")
plt.ylabel("Pixel Count")
plt.grid()
plt.show()

#APPLYING CLAHE TO IMAGE TO SPREAD PIXEL INTENSITIES
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
l_channel_clahe = clahe.apply(l_channel)

#MERGE CHANNELS BACK AND CONVERT TO RGB
lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))
cropped_color_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

#DISPLAY CLAHE ENHANCED IMAGE WITH HISTOGRAM OF ENHANCED IMAGE
plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.title("Clahe Enhanced Image")
plt.imshow(cropped_color_clahe)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.hist(l_channel_clahe.ravel(), 256, range=[0,255], color="black", alpha=0.7)
plt.title("Pixel Intensity Distribution after CLAHE")
plt.xlabel("Intensity Value")
plt.ylabel("Pixel Count")
plt.grid()
plt.show()

#APPLYING WAVELET DENOISING TO IMAGE
import pywt     #import wavelet library

#APPLY WAVELET TO L_CHANNEL OF CLAHE ENHANCED IMAGE
wavelet = pywt.threshold(np.float32(l_channel_clahe)/255.0, 0.1, mode='soft')
l_wavelet = np.uint8(wavelet * 255)

lab_wavelet_denoised = cv2.merge((l_wavelet, a_channel, b_channel))
color_wavelet_denoised = cv2.cvtColor(lab_wavelet_denoised, cv2.COLOR_LAB2RGB)

#DISPLAY WAVELET DENOISED IMAGE
plt.figure(figsize=(12,6))

#ORIGINAL COLOR IMAGE
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cropped_color, cv2.COLOR_BGR2RGB))
plt.title("Original Color Image")
plt.axis("off")

#SHOW WAVELET DENOISED IMAGE
plt.subplot(1, 2, 2)
plt.imshow(color_wavelet_denoised)
plt.title("Wavelet Denoised Image")
plt.axis("off")
plt.show()

#THRESHOLDING FOR SEGMENTING ORGANS
otsu_val, _ = cv2.threshold(l_wavelet, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
bias_val = otsu_val - 15        #bias of 15 (tweak to suit image)
_, otsu_adjusted = cv2.threshold(l_wavelet, bias_val, 255, cv2.THRESH_BINARY)

otsu_segmented = cv2.cvtColor(cv2.merge((otsu_adjusted, a_channel, b_channel)), cv2.COLOR_LAB2RGB)
plt.imshow(otsu_segmented)
plt.title("Organ Segmentation")
plt.axis("off")
plt.show()

# #OPTIONAL (SAVE IMAGES)
# import os

# #CREATING FOLDER FOR IMAGES
# colored_images_paper = "color_output_png"
# os.makedirs(colored_images_paper, exist_ok = True)

# saved_cropped_color_clahe = cv2.cvtColor(cropped_color_clahe, cv2.COLOR_BGR2RGB)
# saved_color_wavelet_denoised = cv2.cvtColor(color_wavelet_denoised, cv2.COLOR_BGR2RGB)
# saved_otsu_segmented = cv2.cvtColor(otsu_segmented, cv2.COLOR_BGR2RGB)

# #SAVING IMAGES
# cv2.imwrite(os.path.join(colored_images_paper, "cropped_image.png"), cropped_color)
# cv2.imwrite(os.path.join(colored_images_paper, "clahe_enhanced.png"), saved_cropped_color_clahe)
# cv2.imwrite(os.path.join(colored_images_paper, "wavelet_denoised.png"), saved_color_wavelet_denoised)
# cv2.imwrite(os.path.join(colored_images_paper, "organ_segmented.png"), saved_otsu_segmented)






