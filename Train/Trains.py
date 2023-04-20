import matplotlib.pyplot as plt
import pandas as pd
import os

from tensorflow import keras


class Train:
    def __init__(self, model, train_ds, valid_ds):
        self.model = model
        self.train_ds = train_ds
        self.valid_ds = valid_ds

    def train(self, epochs, batch_size, save_name):
        print("Training is starting.")
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_psnr',
            patience=20,
            mode='max',
            restore_best_weights=True,
            verbose=1,
        )
        history = self.model.fit(
            self.train_ds,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            validation_data=self.valid_ds,
        )
        self.plot_history(history, save_name)
        print("Training curve is plotted!")
        self.save_history(history, save_name)
        print("Training history is saved!")
        self.save_model(save_name)
        print("Model weights are saved!")

    def plot_history(self, history, save_name):
        print("Training curve is plotting.")
        plt.figure(1)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy/Loss')
        plt.legend(loc='lower right')
        plt.savefig('History/' + save_name + '-training_curve.png')

        plt.figure(2)
        plt.plot(history.history['psnr'], label='psnr')
        plt.plot(history.history['val_psnr'], label='val_psnr')
        # plt.plot(history.history['ssim'], label='ssim')
        # plt.plot(history.history['val_ssim'], label='val_ssim')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.legend(loc='lower right')
        plt.savefig('History/' + save_name + '-psnr_curve.png')

        plt.figure(3)
        plt.plot(history.history['ssim'], label='ssim')
        plt.plot(history.history['val_ssim'], label='val_ssim')
        # plt.plot(history.history['fsim'], label='fsim')
        # plt.plot(history.history['val_fsim'], label='val_fsim')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.legend(loc='lower right')
        plt.savefig('History/' + save_name + '-ssim_curve.png')

    def save_history(self, history, save_name):
        """
        save training accuracy and loss to .csv file
        """
        print("Training history is plotting.")
        acc = pd.Series(history.history['accuracy'], name='accuracy')
        loss = pd.Series(history.history['loss'], name='loss')
        val_acc = pd.Series(history.history['val_accuracy'], name='val_accuracy')
        val_loss = pd.Series(history.history['val_loss'], name='val_loss')
        psnr = pd.Series(history.history['psnr'], name='psnr')
        ssim = pd.Series(history.history['ssim'], name='ssim')
        val_psnr = pd.Series(history.history['val_psnr'], name='val_psnr')
        val_ssim = pd.Series(history.history['val_ssim'], name='val_ssim')
        # fsim = pd.Series(history.history['fsim'], name='fsim')
        # val_fsim = pd.Series(history.history['val_fsim'], name='val_fsim')
        com = pd.concat([acc, val_acc, loss, val_loss, psnr, val_psnr, ssim, val_ssim], axis=1)
        com.to_csv('History/' + save_name + '-log.csv')

    def save_model(self, save_name):
        print("Model weights are saving.")
        json_string = self.model.to_json()
        open(os.path.join('Model/', save_name + '.json'), 'w').write(json_string)
        self.model.save_weights(os.path.join('Model/', save_name + '.hdf5'))


