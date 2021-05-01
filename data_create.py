import cv2
import os
import random
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class datacreate:
    def __init__(self):
        self.mag = 4
        self.num = 0
        self.LR_num = 3

#任意のフレーム数を切り出すプログラム
    def datacreate(self,
                video_path,   #切り取る動画が入ったファイルのpath
                data_number,  #データセットの生成数
                cut_frame,    #1枚の画像から生成するデータセットの数
                cut_height,   #LRの保存サイズ
                cut_width,
                ext='jpg'):

        #データセットのリストを生成
        low_data_list = [[] for _ in range(self.LR_num)]  #LRは3枚で生成。
        high_data_list = []

        video_path = video_path + "/*"
        files = glob.glob(video_path)

        low_cut_height = cut_height // self.mag
        low_cut_width = cut_width // self.mag
    
        while self.num < data_number:
            file_num = random.randint(0, len(files) - 1)
            photo_files = glob.glob(files[file_num] + "/*")
            photo_num = random.randint(0, len(photo_files) - self.LR_num)

            for p in range(cut_frame):
                img = cv2.imread(photo_files[photo_num])
                height, width = img.shape[:2]

                if cut_height > height or cut_width > width:
                    break
            
                ram_h = random.randint(0, height // self.mag - low_cut_height)
                ram_w = random.randint(0, width // self.mag - low_cut_width)

                color_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                gray_img = color_img[:, :, 0]

                low_bi = cv2.resize(gray_img , (int(width // self.mag), int(height // self.mag)), interpolation=cv2.INTER_CUBIC)

                cut_low_bi = low_bi[ram_h : ram_h + low_cut_height, ram_w: ram_w + low_cut_width]
                low_data_list[0].append(cut_low_bi)

                for op in range(self.LR_num - 1):
                    img = cv2.imread(photo_files[photo_num + op + 1])
                    color_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                    gray_img = color_img[:, :, 0]

                    low_bi = cv2.resize(gray_img , (int(width // self.mag), int(height // self.mag)), interpolation=cv2.INTER_CUBIC)

                    cut_low_bi = low_bi[ram_h : ram_h + low_cut_height, ram_w: ram_w + low_cut_width]
                    low_data_list[op + 1].append(cut_low_bi)

                    if op == self.LR_num // 2 - 1:
                        high_data_list.append(gray_img[ram_h * self.mag : ram_h * self.mag + cut_height, ram_w * self.mag : ram_w * self.mag + cut_width])

                self.num += 1

                if self.num == data_number:
                    break

        return low_data_list, high_data_list


                
