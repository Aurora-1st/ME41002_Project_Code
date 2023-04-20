import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input


def vgg19_loss(y_true, y_pred):
    # load pretrained VGG
    vgg19 = VGG19(
        include_top=False,
        input_shape=(None, None, 3),
        weights='imagenet',
    )
    features_extractor = Model(inputs=vgg19.input, outputs=vgg19.get_layer("block2_conv1").output)

    y_true = preprocess_input(y_true * 255.) / 12.75
    y_pred = preprocess_input(y_pred * 255.) / 12.75
    features_pred = features_extractor(y_pred)
    features_true = features_extractor(y_true)

    # adding the scaling factor (to have similar values as with MSE within image space)
    vgg_loss = 0.006 * tf.keras.losses.MeanSquaredError()(features_true, features_pred)
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    # mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    loss = mse + vgg_loss
    return loss
        # tf.math.reduce_mean(tf.math.square(features_pred - features_true), axis=-1)