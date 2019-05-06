from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def mse(imageA,imageB):
    err = np.sum((imageA.astype("float")-imageB.astype("float"))**2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    return err
    
def compare_image(imageA,imageB,title):
    m = mse(imageA,imageB)
    s = ssim(imageA,imageB,multichannel=True)
    p = psnr(imageA,imageB)
    
    '''
    fig = plt.figure(title)
    #plt.suptitle("SSIM : %2f" % s)
    plt.suptitle("MSE : %2f, SSIM : %2f, PSNR: %2f" % (m, s,p))
    
    ax = fig.add_subplot(1,2,1)
    plt.imshow(imageA)
    plt.axis("off")
    
    ax = fig.add_subplot(1,2,2)
    plt.imshow(imageB)
    plt.axis("off")
    
    plt.show()
    '''
    
    return m,s,p
    
dataset_dir = "datasets/Haze2Dehaze/test/B/"
output_dir = "output/B/"


def list_file(data_dir):
    files = []
    for f in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir,f)):
            files.append(os.path.join(data_dir,f))
    
    return files        
        
test_A = list_file(dataset_dir)
output_A = list_file(output_dir)
#print(test_A)
#print(output_A)
ssim_haze = []
psnr_haze = []
mse_haze = []

for (imageA, imageB) in zip(test_A,output_A):
    print(imageA, imageB)
    im1 = cv2.imread(imageA)
    im1 = cv2.resize(im1,(256,256))
    im2 = cv2.imread(imageB)
    im2 = cv2.resize(im2,(256,256))
    
    #compare_image(im1,im2,"Dehaze")
    m,s, p = compare_image(im1,im2,"Dehaze")
    
    ssim_haze.append(s)
    psnr_haze.append(p)
    mse_haze.append(m)
    
    
#print(ssim_haze)        

from statistics import mean   
print("MSE: %2f,SSIM : %2f, PSNR: %2f" % (mean(mse_haze), mean(ssim_haze), mean(psnr_haze)))

'''
imageA = cv2.imread("datasets/Haze2Dehaze/test/B/1401.png",cv2.IMREAD_ANYCOLOR)
cv2.imshow('imageA',imageA)
cv2.waitKey(0)
imageB = cv2.imread("output/B/0001.png",cv2.IMREAD_ANYCOLOR)
cv2.imshow('imageB',imageB)
cv2.waitKey(0)
'''

