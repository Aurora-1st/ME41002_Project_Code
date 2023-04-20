import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers, applications, backend
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd

from Configs.Configuation import Param
from Configs import DataLoader
from Train import Metrics

from image_similarity_measures.quality_metrics import fsim, issm, ssim, psnr
from glob import glob


class DataLoader1:
    def __init__(self, input_size=(96, 96), up_scale=4):
        self.input_size = input_size
        H, W = input_size
        self.out_size = (H * up_scale, W * up_scale)

    def load_data(self, batch_size=1, dataset='train_max_5'):
        path = glob('../Dataset/' + dataset + '/*')

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(contents=img, channels=3)
            img = tf.cast(img, tf.float32)

            img_hr = tf.image.resize(img, self.out_size, method='bicubic')
            img_lr = tf.image.resize(img, self.input_size, method='bicubic')

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr


class ResDenseBlock(layers.Layer):
    def __init__(self, nf=64, gc=32, res_beta=0.2, wd=0.):
        super(ResDenseBlock, self).__init__()
        self.nf = nf
        self.gc = gc
        self.res_beta = res_beta
        self.wd = wd

    def conv2d(self, filters, kernel_size=3, padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=tf.keras.regularizers.l2(5e-4),
               bias_initializer='zeros'):
        return layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_initializer,
        )

    def call(self, x, *args, **kwargs):
        conv1 = self.conv2d(filters=self.gc, kernel_regularizer=tf.keras.regularizers.l2(self.wd))(x)
        conv1 = layers.LeakyReLU(0.2)(conv1)
        conv1 = layers.Concatenate()([x, conv1], )
        conv2 = self.conv2d(filters=self.gc, kernel_regularizer=tf.keras.regularizers.l2(self.wd))(conv1)
        conv2 = layers.LeakyReLU(0.2)(conv2)
        conv2 = layers.Concatenate()([x, conv1, conv2])
        conv3 = self.conv2d(filters=self.gc, kernel_regularizer=tf.keras.regularizers.l2(self.wd))(conv2)
        conv3 = layers.LeakyReLU(0.2)(conv3)
        conv3 = layers.Concatenate()([x, conv1, conv2, conv3])
        conv4 = self.conv2d(filters=self.gc, kernel_regularizer=tf.keras.regularizers.l2(self.wd))(conv3)
        conv4 = layers.LeakyReLU(0.2)(conv4)
        conv4 = layers.Concatenate()([x, conv1, conv2, conv3, conv4])
        conv5 = self.conv2d(filters=self.nf, kernel_regularizer=tf.keras.regularizers.l2(self.wd))(conv4)

        out = conv5 * self.res_beta + x
        return out


class ResInResDenseBlock(layers.Layer):
    def __init__(self, nf=64, gc=32, res_beta=0.2, wd=0., name=None):
        super(ResInResDenseBlock, self).__init__(name=name)
        self.nf = nf
        self.gc = gc
        self.res_beta = res_beta
        self.wd = wd

    def call(self, x, *args, **kwargs):
        out = ResDenseBlock(self.nf, self.gc, self.res_beta, self.wd)(x)
        out = ResDenseBlock(self.nf, self.gc, self.res_beta, self.wd)(out)
        out = ResDenseBlock(self.nf, self.gc, self.res_beta, self.wd)(out)

        out = out * self.res_beta + x
        return out


def conv2d_k3s1(filters, kernel_size=3, strides=1, padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                use_bias=True):
    return layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_bias=use_bias,
    )


def conv2d_k4s2(filters, kernel_size=4, strides=2, padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                use_bias=True):
    return layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_bias=use_bias,
    )


class ESRGAN:
    def __init__(self, input_shape, up_scale=4, num_res_blocks=16):
        """
        SRGAN model build and train.
        """
        self.lr_shape = input_shape
        self.lr_height, self.lr_width, self.channel = input_shape
        self.up_scale = up_scale
        self.hr_height = self.lr_height * up_scale
        self.hr_width = self.lr_width * up_scale
        self.hr_shape = (self.hr_height, self.hr_width, self.channel)
        self.save_name = Param().model_name + '_' + str(Param().times)
        self.dataset_name = 'OralOCTA'

        self.num_res_blocks = num_res_blocks
        optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        self.vgg = self.build_vgg()
        self.vgg.trainable = False

        # 导入数据集
        self.data_loader = DataLoader1()

        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        self.discriminator.summary()

        self.generator = self.build_generator()
        self.generator.summary()

        """combine the generator and discriminator,
        when training, the discriminator is not trained."""
        img_lr = layers.Input(self.lr_shape)

        fake_hr = self.generator(img_lr)
        fake_features = self.vgg(fake_hr)

        self.discriminator.trainable = False
        validity = self.discriminator(fake_hr)
        self.combined = models.Model(img_lr, [validity, fake_features])
        self.combined.compile(
            loss=['binary_crossentropy', 'mse'],
            loss_weights=[5e-1, 1],
            optimizer=optimizer,
            metrics=[Metrics.psnr, Metrics.ssim],
        )

    def build_vgg(self):
        vgg = applications.vgg19.VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=self.hr_shape,
        )
        img_features = [vgg.layers[9].output]
        model = models.Model(vgg.input, img_features)
        return model

    def build_generator(self, nf=64, nb=23, gc=32, wd=0.):
        """
        Residual-in-Residual Dense Block based Model

        :param cfg_net:
        :param gc:
        :param wd:
        :return: generator model
        """

        rrdb_truck_f = ResInResDenseBlock(nf=nf, gc=gc, res_beta=0.2, wd=wd)

        x = layers.Input(self.lr_shape)
        fea = conv2d_k3s1(filters=nf, kernel_regularizer=tf.keras.regularizers.l2(wd))(x)
        fea_rrdb = rrdb_truck_f(fea)
        trunck = conv2d_k3s1(filters=nf, kernel_regularizer=tf.keras.regularizers.l2(wd))(fea_rrdb)
        fea = fea + trunck

        # up-sampling
        size_fea_h = tf.shape(fea)[1] if self.lr_height is None else self.lr_height
        size_fea_w = tf.shape(fea)[2] if self.lr_height is None else self.lr_height
        fea_resize = tf.image.resize(fea, [size_fea_h * 2, size_fea_w * 2], method='nearest')

        fea = conv2d_k3s1(filters=nf, kernel_regularizer=tf.keras.regularizers.l2(wd))(fea_resize)
        fea = layers.LeakyReLU(0.2)(fea)
        fea_resize = tf.image.resize(fea, [size_fea_h * 4, size_fea_w * 4], method='nearest')
        fea = conv2d_k3s1(filters=nf, kernel_regularizer=tf.keras.regularizers.l2(wd))(fea_resize)
        fea = layers.LeakyReLU(0.2)(fea)
        fea = conv2d_k3s1(filters=nf, kernel_regularizer=tf.keras.regularizers.l2(wd))(fea)
        fea = layers.LeakyReLU(0.2)(fea)
        out = conv2d_k3s1(filters=self.channel, kernel_regularizer=tf.keras.regularizers.l2(wd))(fea)

        model = models.Model(x, out)

        return model

    def build_discriminator(self, nf=64, wd=0.):
        """
        Discriminator VGG 128

        :param nf:
        :param wd:
        :return: discriminator model
        """
        inputs = layers.Input(self.hr_shape)

        x = conv2d_k3s1(nf)(inputs)
        x = conv2d_k4s2(nf, use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(0.2)(x)

        x = conv2d_k3s1(nf * 2, use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = conv2d_k4s2(nf * 2, use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(0.2)(x)

        x = conv2d_k3s1(nf * 4, use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = conv2d_k4s2(nf * 4, use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(0.2)(x)

        x = conv2d_k3s1(nf * 8, use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = conv2d_k4s2(nf * 8, use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(0.2)(x)

        x = conv2d_k3s1(nf * 8, use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = conv2d_k4s2(nf * 8, use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(100)(x)
        x = layers.LeakyReLU(0.2)(x)
        out = layers.Dense(1)(x)

        model = models.Model(inputs, out)

        return model

    def scheduler(self, models, epochs):
        global lr
        if epochs % 20000 == 0 and epochs != 0:
            for model in models:
                lr = backend.get_value(model.optimizer.lr)
                backend.set_value(model.optimizer.lr, lr * 0.5)
            print('lr changed to {}'.format(lr * 0.5))

    def train(self, epochs, init_epoch=0, batch_size=1, sample_interval=50):
        save_name = Param().model_name + '_' + str(Param().times)
        # start timing
        start_time = datetime.datetime.now()
        if init_epoch != 0:
            self.generator.load_weights("../Model/weights/%s/gen_epoch%d.h5" % (self.dataset_name, init_epoch),
                                        skip_mismatch=False)
            self.discriminator.load_weights("../Model/weights/%s/dis_epoch%d.h5" % (self.dataset_name, init_epoch),
                                            skip_mismatch=False)

        d_loss_array = []
        g_loss_array = []
        f_loss_array = []
        d_acc_array = []
        g_psnr_array = []

        for epoch in range(init_epoch, epochs):
            # change the learning rate
            self.scheduler([self.combined, self.discriminator], epoch)
            # -------------------- #
            #  train the discriminator
            # ------------------- #
            # load the dataset
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=batch_size)
            print('data loaded.')
            # imgs_hr = self.train_y
            # imgs_lr = self.train_x
            fake_hr = self.generator.predict(imgs_lr)
            print('generator predicted.')

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid * 0.2)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -------------------- #
            # train the generator
            # -------------------- #
            # re-load the dataset，为了打乱顺序
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=batch_size)
            # imgs_hr = self.train_y
            # imgs_lr = self.train_x
            # re-labeled
            valid = np.ones((batch_size,) + self.disc_patch)
            image_features = self.vgg.predict(imgs_hr)

            g_loss = self.combined.train_on_batch(imgs_lr, [valid * 0.2, image_features])
            g_psnr = psnr(imgs_hr, fake_hr)
            print('loss: ', d_loss, g_loss, g_psnr)
            elapsed_time = datetime.datetime.now() - start_time
            print("[Epoch %d] [D loss: %f, acc: %3d%%] [G loss: %05f, feature loss: %05f] time:%s"
                  % (epoch,
                     d_loss[0], 100 * d_loss[1],
                     g_loss[1],
                     g_loss[2],
                     elapsed_time))
            g_loss_array.append(g_loss[1])
            g_psnr_array.append(g_psnr)
            f_loss_array.append(g_loss[2])
            d_loss_array.append(d_loss[0])
            d_acc_array.append(d_loss[1])

            if epoch % sample_interval == 0:
                # 显示图片
                self.sample_images(epoch)
                # 保存图片
                if epoch + 1 % 10000 == 0 and epoch + 1 != init_epoch:
                    os.makedirs('../Model/weights/%s' % self.dataset_name, exist_ok=True)
                    self.generator.save_weights("../Model/weights/%s/gen_epoch%d.h5" % (self.dataset_name, epoch))
                    self.discriminator.save_weights("../Model/weights/%s/dis_epoch%d.h5" % (self.dataset_name, epoch))
        plt.figure(1)
        plt.plot(g_loss_array, label='gen_loss')
        plt.plot(d_loss_array, label='dis_loss')
        plt.plot(f_loss_array, label='feature_loss')
        plt.plot(d_acc_array, label='dis_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy/Loss')
        plt.legend(loc='upper right')
        plt.savefig('../History/' + save_name + '-training_curve.png')

        gen_loss = pd.Series(g_loss_array, name='gen_loss')
        gen_psnr = pd.Series(g_psnr_array, name='gen_psnr')
        dis_loss = pd.Series(d_loss_array, name='dis_loss')
        feature_loss = pd.Series(f_loss_array, name='feature_loss')
        dis_acc = pd.Series(d_acc_array, name='dis_acc')
        com = pd.concat([gen_loss, gen_psnr, dis_loss, feature_loss, dis_acc], axis=1)
        com.to_csv('../History/' + save_name + '-log.csv')

        model = self.generator
        save_name = Param().model_name + '_' + str(Param().times)

        read_single_image(model, save_name)
        read_batch_image(model, save_name)

    def sample_images(self, epoch):
        os.makedirs('../Prediction/' + self.save_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, dataset='train_max_5')
        fake_hr = self.generator.predict(imgs_lr)

        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig('../Prediction/' + self.save_name + '/%d.png' % (epoch))
        plt.close()

        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('../Prediction/' + self.save_name + '/%d_lowers%d.png' % (epoch, i))
            plt.close()
            fig = plt.figure()
            plt.imshow(imgs_hr[i])
            fig.savefig('../Prediction/' + self.save_name + '/%d_highs%d.png' % (epoch, i))
            plt.close()


def read_single_image(model, save_name):
    input_fp = '../Dataset/test_max_5/enface_2022.10.27_15.56.27_oral002lip.oct-Flow_ed.dcm75.jpg'
    save_sr_fp = "../Prediction/" + save_name + "_sr_out.jpg"
    save_hr_fp = "../Prediction/hr_out.jpg"
    save_lr_fp = "../Prediction/lr_out.jpg"

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
    print(pred_super_resolution.shape)
    pred_super_resolution = tf.squeeze(pred_super_resolution, axis=0)

    origin_high_resolution = origin_high_resolution.numpy()
    pred_super_resolution = pred_super_resolution.numpy()

    PSNR = psnr(origin_high_resolution, pred_super_resolution, max_p=1)
    SSIM = ssim(origin_high_resolution, pred_super_resolution, max_p=1)
    FSIM = fsim(origin_high_resolution, pred_super_resolution)
    ISSM = issm(origin_high_resolution, pred_super_resolution)
    NQM = Metrics.nqm(origin_high_resolution, pred_super_resolution)

    print('psnr: ', PSNR)
    print('ssim: ', SSIM)
    print('fsim: ', FSIM)
    print('issm: ', ISSM)

    tf.keras.preprocessing.image.save_img(save_sr_fp, pred_super_resolution)
    # keras.preprocessing.image.save_img(save_hr_fp, origin_high_resolution)
    # keras.preprocessing.image.save_img(save_lr_fp, degraded_low_resolution)


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
        predicted = tf.squeeze(predicted, axis=0)
        predicted = predicted.numpy()
        predicted_list.append(predicted)
    n_img = len(predicted_list)
    dirname = '../Prediction/'

    for img in test_y:
        img = tf.cast(img, dtype=tf.float32)
        # print(img.shape)
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
    ## use eager mode to train the network if necessary
    # tf.config.experimental_run_functions_eagerly(True)
    gan = ESRGAN(input_shape=Param().img_shape, up_scale=4, num_res_blocks=16)
    gan.train(epochs=4000, batch_size=1, sample_interval=500)
