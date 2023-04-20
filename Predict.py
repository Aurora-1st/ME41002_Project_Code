from Configs.Configuation import Param
from Configs import DataLoader
from Network import Networks, DenseSTSR_v3
from Train import Metrics
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd

from quality_metrics import fsim, issm, ssim, psnr


def main():
    model = DenseSTSR_v3.PublicDenseSTSR(Param().img_shape)()
    save_name = Param().model_name + '_' + str(Param().times)
    model.load_weights(os.path.join('Model/', save_name + '.hdf5'))

    read_single_image(model, save_name)
    read_batch_image(model, save_name)
    # read_single_image(save_name)
    # read_batch_image(save_name)


def read_single_image(model, save_name):
    # input_fp = 'Dataset/test_max_5/enface_2022.10.27_15.56.27_oral002lip.oct-Flow_ed.dcm75.jpg'
    input_fp = 'Dataset/valid_max_5/enface_2022.10.31_11.08.06_oral003lip.oct-Flow_ed.dcm55.jpg'
    save_sr_fp = "Prediction/" + save_name + "_sr_out_2.jpg"
    save_hr_fp = "Prediction/hr_out_2.jpg"
    save_lr_fp = "Prediction/lr_out_2.jpg"

    image = tf.io.read_file(input_fp)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)

    origin_high_resolution = tf.image.resize(
        images=image,
        size=(384, 384),
        method='bicubic')
    origin_high_resolution = origin_high_resolution / 255.

    degraded_low_resolution = tf.image.resize(
        images=image,
        size=(96, 96),
        method='bicubic')
    degraded_low_resolution = degraded_low_resolution / 255.

    image = tf.image.resize(images=image,
                            size=(96, 96),
                            method='bicubic')
    image = image / 255.
    image = tf.expand_dims(image, axis=0)

    pred_super_resolution = model(image)
    # pred_super_resolution = tf.image.resize(
    #     images=image,
    #     size=(384, 384),
    #     method='bicubic',
    # )

    print(pred_super_resolution.shape)
    pred_super_resolution = tf.squeeze(pred_super_resolution, axis=0)

    pred_super_resolution = pred_super_resolution.numpy()
    origin_high_resolution = origin_high_resolution.numpy()

    PSNR = psnr(origin_high_resolution, pred_super_resolution, max_p=1)
    SSIM = ssim(origin_high_resolution, pred_super_resolution, max_p=1)
    FSIM = fsim(origin_high_resolution, pred_super_resolution)
    ISSM = issm(origin_high_resolution, pred_super_resolution)
    NQM = Metrics.nqm(origin_high_resolution, pred_super_resolution)

    print('psnr: ', PSNR)
    print('ssim: ', SSIM)
    print('fsim: ', FSIM)
    print('issm: ', ISSM)
    print('nqm: ', NQM)

    keras.preprocessing.image.save_img(save_sr_fp, pred_super_resolution)
    keras.preprocessing.image.save_img(save_hr_fp, origin_high_resolution)
    keras.preprocessing.image.save_img(save_lr_fp, degraded_low_resolution)


def read_batch_image(model, save_name):
    test_x = DataLoader.DataLoader(
        (96, 96), 3, img_batches=Param().num_test_data,
        file_path=Param().LR_test_path).return_tensor()  # input
    test_y = DataLoader.DataLoader(
        (384, 384), 3, img_batches=Param().num_test_data,
        file_path=Param().HR_test_path).return_tensor()  # ground truth

    predicted_list = []
    true_list = []
    PSNR_array = []
    SSIM_array = []
    FSIM_array = []
    ISSM_array = []
    NQM_array = []

    for img in test_x:
        img = tf.cast(img, dtype=tf.float32)
        img = tf.expand_dims(img, axis=0)
        predicted = model(img)
        # predicted = tf.image.resize(
        #     images=img,
        #     size=(384, 384),
        #     method='bicubic',
        # )
        predicted = tf.squeeze(predicted, axis=0)
        predicted = predicted.numpy()
        predicted_list.append(predicted)
    n_img = len(predicted_list)
    dirname = 'Prediction/'

    for img in test_y:
        img = tf.cast(img, dtype=tf.float32)
        img = img.numpy()
        true_list.append(img)

    for i in range(n_img):
        PSNR = psnr(true_list[i], predicted_list[i], max_p=1)
        SSIM = ssim(true_list[i], predicted_list[i], max_p=1)
        FSIM = fsim(true_list[i], predicted_list[i])
        ISSM = issm(true_list[i], predicted_list[i])
        NQM = Metrics.nqm(true_list[i], predicted_list[i])

        print("_psnr:", PSNR)
        print("_ssim:", SSIM)
        print("_fsim:", FSIM)
        print("_issm:", ISSM)
        print("_nqm:", NQM)

        PSNR_array.append(PSNR)
        SSIM_array.append(SSIM)
        FSIM_array.append(FSIM)
        ISSM_array.append(ISSM)
        NQM_array.append(NQM)

    PSNR_mean = tf.math.reduce_mean(PSNR_array)
    SSIM_mean = tf.math.reduce_mean(SSIM_array)
    FSIM_mean = tf.math.reduce_mean(FSIM_array)
    ISSM_mean = tf.math.reduce_mean(ISSM_array)
    NQM_mean = tf.math.reduce_mean(NQM_array)

    mean = [PSNR_mean, SSIM_mean, FSIM_mean, ISSM_mean, NQM_mean]
    print(mean)

    PSNR_std = tf.math.reduce_std(PSNR_array)
    SSIM_std = tf.math.reduce_std(SSIM_array)
    FSIM_std = tf.math.reduce_std(FSIM_array)
    ISSM_std = tf.math.reduce_std(ISSM_array)
    NQM_std = tf.math.reduce_std(NQM_array)

    std = [PSNR_std, SSIM_std, FSIM_std, ISSM_std, NQM_std]
    print(std)
    index = ['PSNR', 'SSIM', 'FSIM', 'ISSM', 'NQM']

    dataframe = pd.DataFrame({'index': index, 'mean': mean, 'std': std})

    dataframe.to_csv(
        dirname + save_name + "_metrics.csv",
        index=True,
        sep=',',
    )


if __name__ == '__main__':
    main()
