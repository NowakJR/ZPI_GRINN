'''
wyciecie male fragmentu do porownania
'''
from PIL import Image
import numpy as np
import os
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
import cv2
ilosc_plytek = 9


def zrob_folder(x=2):
    for i in range(x):
        os.mkdir(f'C:/Users/szymo/Desktop/ZPI - AOI/ZPI_GRINN/szymonek/zadanie od dimy/fragmenty/plytka{i + 1}')


# zrob_folder(ilosc_plytek)
# with open("kordy_czesci.txt") as plik:
#     dane = []
#     for i in plik:
#         wiersz = i.split()
#         dane.append(list(map(int, wiersz)))

dane = dict() #1,2,7,8 no deffects
# 3 - deffect 2
# 4 - deffect 3
# 5 - deffect 4
# 6 - deffect 5
# 9 - deffect 6
dane[1] = [[203, 539, 68, 30]  #
    , [1057, 628, 37, 23]  # 1094, 651
    , [1056, 604, 41, 23]  # 1097, 627
    , [1412, 761, 64, 35]  # 1476, 796
    , [1429, 238, 45, 50]]  # 1474, 288

dane[2] = [[206, 538, 68, 30]
    , [1065, 631, 37, 23]
    , [1065, 606, 41, 23]
    , [1415, 760, 64, 35]
    , [1434, 240, 45, 50]
           ]

dane[3] = [[201, 541, 68, 30]
    , [1057, 623, 37, 23]
    , [1054, 603, 41, 23]
    , [1410, 754, 64, 35]
    , [1425, 235, 45, 50]
           ]

dane[4] = [[209, 543, 68, 30]
    , [1065, 627, 37, 23]
    , [1065, 602, 41, 23]
    , [1427, 751, 64, 35]
    , [1431, 230, 45, 50]]

dane[5] = [[203, 558, 68, 30]
    , [1055, 638, 37, 23]
    , [1055, 613, 41, 23]
    , [1409, 762, 64, 35]
    , [1421, 244, 45, 50]]

dane[6] = [[205, 539, 68, 30]
    , [1059, 625, 37, 23]
    , [1062, 602, 41, 23]
    , [1420, 751, 64, 35]
    , [1429, 228, 45, 50]]

dane[7] = [[232, 540, 68, 30]
    , [1084, 629, 37, 23]
    , [1086, 606, 41, 23]
    , [1443, 760, 64, 35]
    , [1459, 240, 45, 50]]

dane[8] = [[208, 542, 68, 30]
    , [1065, 621, 37, 23]
    , [1063, 598, 41, 23]
    , [1416, 746, 64, 35]
    , [1428, 228, 45, 50]]

dane[9] = [[192, 540, 68, 30]
    , [1053, 622, 37, 23]
    , [1054, 596, 41, 23]
    , [1411, 746, 64, 35]
    , [1422, 228, 45, 50]]

# slownik_plytek = dict.fromkeys([x + 1 for x in range(9)], [])
# list1, list2, list3, list4, list5, list6, list7, list8 = ([] for i in range(8))
pcb_list = [[] for i in range(9)]

for i in range(9):
    im1 = cv2.imread(f'detailcrop/simplerot_crop{i + 1}.jpg')
    for j in dane[i+1]:
        left, top, right, bottom = j
        right += left
        bottom += top
        im1c = im1[top:bottom, left:right]
        # slownik_plytek[i + 1].append(j[0])
        pcb_list[i].append(im1c)
        # im1c.show()
        # im1c.save(
        #     f'C:/Users/szymo/Desktop/ZPI - AOI/ZPI_GRINN/szymonek/zadanie od dimy/fragmenty/plytka{i+1}/czesc{dane.index(j)}.jpeg',
        #     'JPEG')
        # print(slownik_plytek)


def check_algorithms():
    k = 0
    # cv2.imshow('img', pcb_list[k+2][k])
    # cv2.waitKey()
    before = pcb_list[0][k]
    rmse_list = []
    ergas_list = []
    rase_list = []
    for i in range(9):
        print("Płytka nr", i+1)
        after = pcb_list[i][k]
        print("MSE: ", mse(before, after))
        print("RMSE: ", rmse(before, after)) #git
        print("PSNR: ", psnr(before, after))
        print("SSIM: ", ssim(before, after))
        print("UQI: ", uqi(before, after))
        print("ERGAS: ", ergas(before, after)) #git
        print("SCC: ", scc(before, after))
        print("RASE: ", np.int(rase(before, after))) #git
        print("SAM: ", sam(before, after))
        rmse_rate = rmse(before, after)
        ergas_rate = ergas(before, after)
        rase_rate = rase(before, after)

        rmse_list.append(rmse_rate)
        ergas_list.append(ergas_rate)
        rase_list.append(rase_rate)

    rmse_max = rmse_list.index(max(rmse_list))+1
    ergas_max = ergas_list.index(max(ergas_list))+1
    rase_max = rase_list.index(max(rase_list))+1

    print(rmse_max, ergas_max, rase_max)

# rmse_list = []
# ergas_list = []
# rase_list = []
# for golden_i in [0, 1, 6, 7]:
#     part = 4
#     before = pcb_list[golden_i][part]
#     rmse_list = []
#     ergas_list = []
#     rase_list = []
#     for i in range(9):
#         after = pcb_list[i][part]
#         cv2.imshow('img', pcb_list[i][part])
#         cv2.waitKey()
#         rmse_rate = rmse(before, after)
#         ergas_rate = ergas(before, after)
#         rase_rate = rase(before, after)
#
#         rmse_list.append(rmse_rate)
#         ergas_list.append(ergas_rate)
#         rase_list.append(rase_rate)
#     rmse_max = rmse_list.index(max(rmse_list)) + 1
#     ergas_max = ergas_list.index(max(ergas_list)) + 1
#     rase_max = rase_list.index(max(rase_list)) + 1
#     print(rmse_max, ergas_max, rase_max)

check_algorithms()