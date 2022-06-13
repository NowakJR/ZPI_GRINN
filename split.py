import cv2
import sys
import os



def split_image(img, nRows, mCols):
    # Dimensions of the image
    sizeX = img.shape[1]
    sizeY = img.shape[0]
    roi_list = []
    for i in range(0,nRows):
        for j in range(0, mCols):
            roi = img[int(i*sizeY/nRows):int(i*sizeY/nRows + sizeY/nRows) ,int(j*sizeX/mCols):int(j*sizeX/mCols + sizeX/mCols)]
            # cv2.imshow('rois'+str(i)+str(j), roi)
            # cv2.imwrite('patches/patch_'+str(i)+str(j)+".jpg", roi)
            roi_list.append(roi)
    # cv2.waitKey()
    return roi_list

if __name__ == "__main__":
    if not os.path.exists('patches'):
        os.makedirs('patches')

    img = cv2.imread('detailcrop/simplerot_crop1.jpg')
    nRows = 10
    mCols = 10
    row_list = split_image(img, nRows, mCols)