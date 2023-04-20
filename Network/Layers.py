from tensorflow.keras import layers, models


class SRCNN:
    def __init__(self, img_shape=(96, 96, 1)):
        self.dim = img_shape[-1]
        self.img_shape = img_shape

    def __call__(self):
        inputs = layers.Input(shape=self.img_shape)
        x = layers.UpSampling2D(4)(inputs)

        conv1 = layers.Conv2D(64, 9, padding='same', name='conv1')(x)
        relu1 = layers.Activation('relu', name='relu1')(conv1)
        conv2 = layers.Conv2D(32, 1, padding='same', name='conv2')(relu1)
        relu2 = layers.Activation('relu', name='relu2')(conv2)
        out = layers.Conv2D(self.dim, 5, padding='same', name='conv3')(relu2)

        model = models.Model(inputs=inputs, outputs=out, name="SRCNN")
        return model


class FSRCNN:
    def __init__(self, img_shape=(96, 96, 1)):
        self.dim = img_shape[-1]
        self.img_shape = img_shape

    def conv2d(self, filters, kernel_size, padding='same', kernel_initializer='he_normal'):
        return layers.Conv2D(
            filters=filters, kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=kernel_initializer,
        )

    def __call__(self):
        inputs = layers.Input(shape=self.img_shape)

        x = self.conv2d(64, 5)(inputs)
        x = layers.PReLU()(x)

        x = self.conv2d(12, 1)(x)
        x = layers.PReLU()(x)

        x = self.conv2d(12, 3)(x)
        x = layers.PReLU()(x)
        x = self.conv2d(12, 3)(x)
        x = layers.PReLU()(x)
        x = self.conv2d(12, 3)(x)
        x = layers.PReLU()(x)
        x = self.conv2d(12, 3)(x)
        x = layers.PReLU()(x)

        x = self.conv2d(64, 1)(x)
        x = layers.PReLU()(x)

        out = layers.Conv2DTranspose(
            filters=self.dim,
            kernel_size=9,
            strides=4,
            padding='same',
        )(x)

        model = models.Model(inputs=inputs, outputs=out)

        return model


class VDSR:
    def __init__(self, img_shape=(96, 96, 1)):
        self.img_shape = img_shape
        self.dim = img_shape[-1]

    def __call__(self, *args, **kwargs):
        input_img = layers.Input(shape=self.img_shape)
        low_resolution = layers.UpSampling2D(4)(input_img)

        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(low_resolution)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)

        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)

        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)

        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        model = layers.Activation('relu')(model)
        model = layers.Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
        res_img = model

        high_resolution = layers.add([res_img, low_resolution])

        model = models.Model(input_img, high_resolution)

        return model

