from split import split_image
import cv2 as cv2
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('detailcrop/simplerot_crop1.jpg')
img2 = cv2.imread('detailcrop/simplerot_crop_new2_more_el.jpg')
# img2 = cv2.imread('detailcrop/simplerot_crop4.jpg')
# img2 = cv2.imread('detailcrop/simplerot_crop1_2pix.jpg')
# img2 = cv2.imread('detailcrop/simplerot_crop1_copy.jpg')

nRows = 220
mCols = 200


# img2 = cv2.blur(img2, (3,3))
# img2 = cv2.GaussianBlur(img2, (5,5), 1,1) #git
# img2 = cv2.bilateralFilter(img2,9,220,75)  #best


roi_list1 = split_image(img1, nRows, mCols)
roi_list2 = split_image(img2, nRows, mCols)




diff_array = np.zeros((nRows, mCols))
c = 0
for i in range(nRows):
    for j in range(mCols):
        diff_array[i][j] = int(rmse(roi_list1[c], roi_list2[c]))
        # diff_array[i][j] = np.int(rase(roi_list1[c], roi_list2[c]))

        c+=1
print(np.argmax(diff_array))
a = [i for i in range(20)]
print(a)
cols = np.argmax(diff_array, axis=0)
rows = np.argmax(diff_array, axis=1)
print(cols)

# roi_list = np.reshape(roi_list1, (nRows, mCols))
# for i in a:
#     cv2.imshow('img', roi_list1[727])
#     cv2.waitKey()

plt.pcolormesh(diff_array, cmap = 'cool')
plt.show()
