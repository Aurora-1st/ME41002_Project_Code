from Configs.Configuation import Param
from Configs import DataLoader
from Network import Networks
from Network import SwinIR, DenseSTSR_v3
from Train import Trains
from Train import Metrics, LossFunc
from image_similarity_measures.quality_metrics import ssim, psnr

import tensorflow as tf
import os


def main():
    train_x = DataLoader.DataLoader((96, 96), 3, img_batches=Param().num_train_data,
                                    file_path=Param().LR_train_path).return_tensor()  # LR
    train_y = DataLoader.DataLoader((384, 384), 3, img_batches=Param().num_train_data,
                                    file_path=Param().HR_train_path).return_tensor()  # HR
    valid_x = DataLoader.DataLoader((96, 96), 3, img_batches=Param().num_valid_data,
                                    file_path=Param().LR_valid_path).return_tensor()  # LR
    valid_y = DataLoader.DataLoader((384, 384), 3, img_batches=Param().num_valid_data,
                                    file_path=Param().HR_valid_path).return_tensor()  # HR
    train_ds = DataLoader.zip_and_batch_ds(train_x, train_y, Param().batch_size)
    valid_ds = DataLoader.zip_and_batch_ds(valid_x, valid_y, Param().batch_size)
    print("Training set is loaded!")

    save_name = Param().model_name + '_' + str(Param().times)
    model = DenseSTSR_v3.PublicDenseSTSR(
        image_shape=Param().img_shape)()

    print("Model is built!")
    model.compile(
        optimizer=Param().optimizer,
        loss=Param().loss,
        metrics=[Metrics.psnr, Metrics.ssim, 'accuracy'],
    )
    model.summary()

    Trains.Train(model, train_ds, valid_ds).train(
        epochs=Param().epochs,
        batch_size=Param().batch_size,
        save_name=save_name,
    )
    print("Training is finished!")


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices(gpus[0], 'GPU')

    ## use eager mode to train the model if necessary
    # tf.config.run_functions_eagerly(True)

    main()
