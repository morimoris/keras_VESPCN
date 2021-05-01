import model
import data_create
import argparse
import os
import cv2
import glob
import keras
import tensorflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

    train_height = 160
    train_width = 160
    test_height = 720 
    test_width = 1280

    train_dataset_num = 20000
    test_dataset_num = 10
    train_cut_num = 10
    test_cut_num = 1

    train_movie_path = "../../reds/train_sharp"
    test_movie_path = "../../reds/val_sharp"

    input_LR_num = 3
    input_channels = 1
    mag = 4

    MAX_BATSH_SIZE = 128
    EPOCHS = 1000
    
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, 1, name=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_model', help='train_datacreate, test_datacreate, train_model, evaluate')

    args = parser.parse_args()

    if args.mode == 'train_datacreate':
        datacreate = data_create.datacreate()
        train_x, train_y = datacreate.datacreate(train_movie_path,   #切り取る動画のpath
                                            train_dataset_num,       #データセットの生成数
                                            train_cut_num,           #1枚の画像から生成するデータの数
                                            train_height,            #保存サイズ
                                            train_width)   
        path = "train_data_list"
        np.savez(path, train_x, train_y)

    elif args.mode == 'test_datacreate':
        datacreate = data_create.datacreate()
        test_x, test_y = datacreate.datacreate(test_movie_path,
                                            test_dataset_num,
                                            test_cut_num,
                                            test_height,
                                            test_width)

        path = "test_data_list"
        np.savez(path, test_x, test_y)

    elif args.mode == "train_model":
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
        else:
            print("Not enough GPU hardware devices available")

        npz = np.load("train_data_list.npz")
        train_x = npz["arr_0"]
        train_y = npz["arr_1"]

        train_x = tf.convert_to_tensor(train_x, np.float32)
        train_y = tf.convert_to_tensor(train_y, np.float32)

        train_x /= 255
        train_y /= 255

        train_model = model.VESPCN()

        optimizers = tf.keras.optimizers.Adam(learning_rate=1e-4)
        train_model.compile(loss = "mean_squared_error",
                        optimizer = optimizers,
                        metrics = [psnr])

        x = MAX_BATSH_SIZE
        i = 0
        while x > 1:
            x /= 2
            i += 1

        for n in range(i):
            train_model.fit({"input_t_minus_1":train_x[0], "input_t":train_x[1], "input_t_plus_1":train_x[2]},
                            train_y,
                            epochs = 10,
                            verbose = 2,
                            batch_size = 2 ** n)

        train_model.fit({"input_t_minus_1":train_x[0], "input_t":train_x[1], "input_t_plus_1":train_x[2]},
                        train_y,
                        epochs = EPOCHS - 10 * i,
                        verbose = 2,
                        batch_size = MAX_BATSH_SIZE)

        train_model.save("VESPCN_model.h5")

    elif args.mode == "evaluate":
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
        else:
            print("Not enough GPU hardware devices available")

        result_path = "result"
        os.makedirs(result_path, exist_ok = True)

        npz = np.load("test_data_list.npz", allow_pickle = True)

        test_x = npz["arr_0"]
        test_y = npz["arr_1"]

        test_x = tf.convert_to_tensor(test_x, np.float32)
        test_y = tf.convert_to_tensor(test_y, np.float32)

        test_x /= 255
        test_y /= 255
            
        path = "VESPCN_model.h5"

        if os.path.exists(path):
            model = tf.keras.models.load_model(path, custom_objects={'psnr':psnr})
            pred = model.predict({"input_t_minus_1":test_x[0], "input_t":test_x[1], "input_t_plus_1":test_x[2]}, batch_size = 1)

            ps_pred_ave = 0
            ps_low_ave = 0

            for p in range(len(test_y)):
                pred[p][pred[p] > 1] = 1
                pred[p][pred[p] < 0] = 0
                ps_pred = psnr(tf.reshape(test_y[p], [test_height, test_width, 1]), pred[p])
                    
                ps_pred_ave += ps_pred

                if True:
                    low_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(test_x[1][p] * 255, [test_height // mag, test_width // mag]))
                    cv2.imwrite(result_path + "/" + str(p) + "_low" + ".jpg", low_img) #LR

                    low_4 = tf.keras.preprocessing.image.img_to_array(tf.reshape(test_x[1][p] * 255, [test_height // mag, test_width // mag]))
                    low_4 = cv2.resize(low_4 , (int(test_width), int(test_height)), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(result_path + "/" + str(p) + "_low_4" + ".jpg", low_4) #LR*4

                    high_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(test_y[p] * 255, [test_height, test_width]))
                    cv2.imwrite(result_path + "/" + str(p) + "_high" + ".jpg", high_img)   #HR

                    pred_img = tf.keras.preprocessing.image.img_to_array(tf.reshape(pred[p] * 255, [test_height, test_width]))
                    cv2.imwrite(result_path + "/" + str(p) + "_pred" + ".jpg", pred_img)

                    print("num:{}".format(p))
                    print("psnr_pred:{}".format(ps_pred))

            print("psnr_pred_average:{}".format(ps_pred_ave / len(test_y)))


  
 
