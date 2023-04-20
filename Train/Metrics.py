import tensorflow as tf


def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def psnr(y_true, y_pred):
    return - 10 * tf.math.log(mse(y_true, y_pred))/tf.math.log(10.)


def ssim(y_true, y_pred):
    mu_x = tf.math.reduce_mean(y_true)
    var_x = tf.math.reduce_std(y_true)
    mu_y = tf.math.reduce_mean(y_pred)
    var_y = tf.math.reduce_std(y_pred)
    cov = tf.math.reduce_mean((y_true - mu_x) * (y_pred - mu_y))
    c1 = tf.square(0.01)
    c2 = tf.square(0.03)
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))


def nqm(y_true, y_pred):
    return tf.reduce_mean(10 * tf.math.log(tf.square(y_pred)/tf.square(y_true - y_pred)) / tf.math.log(10.))

