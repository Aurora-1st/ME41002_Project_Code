import tensorflow as tf
from tensorflow.keras import layers, initializers
import numpy as np


class TruncatedDense(layers.Dense):
    def __init__(self, units, use_bias=True,
                 initializer=initializers.TruncatedNormal(mean=0., stddev=.02)):
        super(TruncatedDense, self).__init__(
            units,
            use_bias=use_bias,
            kernel_initializer=initializer,
        )


class MLP(layers.Layer):
    """
    Multi-layer perception
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=layers.Activation('gelu'), drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = TruncatedDense(units=hidden_features)
        self.act = act_layer
        self.fc2 = TruncatedDense(units=out_features)
        self.drop = layers.Dropout(drop)

    def call(self, inputs, training=None, mask=None):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


def window_partition(x, window_size=2):
    """ Window partition function for the Shift-window Vision Transformer.

    Split the input tensor from: [height, width, channels] to ==>>
    [height/window_size, window_size, width/window_size, window_size, channels]

    Then, transpose the order of the tensor, to ==>>
    [height/window_size, width/window_size, window_size, window_size, channels]

    Finally, reshape the tensor to ==>>
    [height/window_size*width/window_size, window_size, window_size, channels]

    """
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(tensor=x, shape=(
        -1, patch_num_y, window_size, patch_num_x, window_size, channels))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    """ Window reverse function for the Shift-window Vision Transformer.

    Convert the tensor from previous divided windows to normal shape, from
    [height/window_size*width/window_size, window_size, window_size, channels]
    to: => [height, width, channels]

    """
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(tensor=windows, shape=(
        -1, patch_num_y, patch_num_x, window_size, window_size, channels))
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


@tf.keras.utils.register_keras_serializable()
class WindowSelfAttention(tf.keras.layers.Layer):
    """ Window MultiHead Self Attention Layer.
    For example, the input is from the example tensor in WinMHSABlocks. And the
    shape of the input tensor is: (Batch Size*(64/4)*(64/4), 4*4, 96)
    a). use the dense layer with hidden_size*3 units to generate the qkv tensor.
    The output is called qkv with shape=(Batch Size*(64/4)*(64/4), 4*4, 96*3)
    b). assume that the num_heads is set as 3, and the qkv tensor from (a) is
    reshaped to -->> (3, Batch Size*(64/4)*(64/4), 3, 4*4, 32)
    c). then, divide the qkv tensor to q, k, v tensor, respectively. And hence
    each q, k, v tensor has shape: (Batch Size*(64/4)*(64/4), 3, 4*4, 32). And
    represent that each q/k/v tensor has 4096 windows, and each windows with the
    3 different heads, 4*4 patches and 32 dimensions feature.

    d). layer normalization for the tensor -> q
    e). then, q @ k as the first-step of the attention layer, and the shape
    of the q @ k outputs is: (Batch Size*(64/4)*(64/4), 3, 4*4, 4*4)

    """

    # TODO: Fully understand window-based self-attention
    def __init__(self, hidden_size, window_size, num_heads=6, drop_prob=0.03,
                 drop_rate=0.3, use_bias=True, **kwargs):
        super(WindowSelfAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.drop_rate = drop_rate
        self.use_bias = use_bias
        self.scale = (hidden_size // num_heads) ** -0.5

        self.qkv_dense_layer = layers.Dense(units=hidden_size * 3, use_bias=use_bias)
        self.dropout_layer = layers.Dropout(drop_rate)
        self.projection_dense_layer = layers.Dense(hidden_size)

    def build(self, input_shape):
        """ Obtain the relative position bias information.
        For example, the window size is set as 3 in this introduction.
        a). generate the 'coords' with shape 2 * 3 * 3. It means that in the
        window with size 3 * 3, each position has its coordinates [x, y],
        and the relative_coords with shape 2 * 9 * 9, it means in the 9 points,
        each points [x, y] and other points have the error. For example,
        [0][3][1] is represent the error difference between No.3 points and
        No.1 points. Then reshape, and let those two coordinates add the
        value in 3-1=2. That is because the range of those coords is [0, 2].
        Then, let the coords-y multiply the (2*3-1=5). Finally, add the error
        difference value in x,y dimensions, and obtain the
        relative_position_index, 3^2 * 3^2, are the relative position bias
        between two points. finally, find the position in bias_table.
        b). add the relative_position to the attention output, and apply the
        softmax on it.
        c). ((q @ k) + relative_position)/sqrt(dims) @ v. Then the output shape
        is: (Batch Size*(64/4)*(64/4), 3, 4*4, 96)
        """
        num_window_elements = (2 * self.window_size[0] - 1) * (
                2 * self.window_size[1] - 1)  # (2*2-1) * (2*2-1) = 9
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
            name="name_for_run" +
                 str(np.random.rand(1000)[np.random.randint(1000)]),
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None,
                                                       :]
        relative_coords = relative_coords.transpose(
            [1, 2, 0])  # shape from (2, 4, 4) ==>> (4, 4, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(  # (4, 4) matrix
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
            name="name_for_run" +
                 str(np.random.rand(1000)[np.random.randint(1000)])
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dims = channels // self.num_heads
        qkv = self.qkv_dense_layer(x)
        qkv = tf.reshape(qkv, shape=(-1, size, 3, self.num_heads, head_dims))
        qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,))
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat)
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements,
                                           num_window_elements, -1))
        relative_position_bias = tf.transpose(relative_position_bias,
                                              perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            # mask shape: (1024, 4, 1)
            # after expand_dims: (1, 1024, 1, 4, 1)
            mW = mask.get_shape()[0]
            mask_float = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1),
                                                axis=0), dtype=tf.float32)
            # attn after reshape: (BS*, 1024, 4, 64, 64)
            attn = (tf.reshape(attn,
                               shape=(-1, mW, self.num_heads, size, size)) +
                    mask_float)
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = tf.keras.activations.softmax(attn, axis=-1)
        else:
            attn = tf.keras.activations.softmax(attn, axis=-1)

        attn = self.dropout_layer(attn)

        attn_output = attn @ v
        attn_output = tf.transpose(attn_output, perm=(0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, shape=(-1, size, channels))
        attn_output = self.projection_dense_layer(attn_output)
        attn_output = self.dropout_layer(attn_output)

        return attn_output

    def get_config(self):
        config = super(WindowSelfAttention, self).get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "drop_prob": self.drop_prob,
            "drop_rate": self.drop_rate,
            "use_bias": self.use_bias,
            "scale": self.scale,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ShiftWindowsTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_patches, num_heads, window_size=7,
                 shift_size=0, mlp_units=1024, use_bias=True, drop_rate=0.0,
                 block="0", stage="0", **kwargs):
        """ When the shift_size == 0, this is a window-MSA block:
        For example, the input shape is (Batch size, 64*64, 96), H=W=64
        a). the input will be first processed by layer-normalization.
        b). then, reshape to the (Batch Size, 64, 64, 96).
        c). use the defined 'function' -- 'window_partition' to divide the
        tensor from (Batch Size, 64, 64, 96) to ==>> ("assume window size = 4")
        (Batch Size*(64/4)*(64/4), 4, 4, 96)
        d). reshape the output from (c) -->> (Batch Size*(64/4)*(64/4), 4*4, 96)
        e). finally, use the WindowAttention Layer to process the tensor. For
        the shift_size==0, the attn_mask is set as 'None'.

        f). ....

        """
        super(ShiftWindowsTransformerBlock, self).__init__(
            name=("SwinTransformerBasicLayer" +
                  "block_" + str(block) + "_stage_" + str(stage)), **kwargs)
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.layer_normalization_1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowSelfAttention(
            hidden_size=hidden_size, window_size=(window_size, window_size),
            num_heads=num_heads, use_bias=use_bias, drop_rate=drop_rate,
        )
        self.drop_path = layers.Dropout(0.05)
        self.layer_normalization_2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp_block = MLP(in_features=hidden_size, hidden_features=hidden_size)

        if min(self.num_patches) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patches)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            """ Create the attn_mask
            assume the input shape is (8^2*B, 7^2, 96), the input tensor for the
            attention layer. also, assume the num_heads is set as 3. Hence, the 
            input paras of the attention layer is 'x_windows' and 'attn_mask'.

            To generate the attn_mask:
            a) the shape of the mask should be (1, 56, 56, 1), which is the same
            as the input image shape (i.e., width and height of feature maps). 
            b) based on mask, generate the h_slices and w_slices, which with 
            value (0, -7), (-7, -3), (-3, None), after the recycle processing, 
            the mask will be divided into 9 patches and be marked. 

            Example of Mask for better understanding: 
            12 x 12 patches, window have 3 x 3 patches, shift size = 1, 4 x 4 windows.


            """
            height, width = self.num_patches
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1

            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(tensor=mask_windows, shape=[
                -1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=-1) - tf.expand_dims(
                mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            # shape: (height/window_size*width/window_size, window_size^2, 1)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                trainable=False,
                name="name_for_run" +
                     str(np.random.rand(1000)[np.random.randint(1000)])
            )

    def call(self, x):
        height, width = self.num_patches
        _, num_patches_before, channels = x.shape
        x_skip = x

        x = self.layer_normalization_1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            """ Reason of using the tf.roll when shift_size > 0.
            The disadvantage of the Window-MHSA is that the information from 
            the different divided patches cannot share with other patches. 
            Hence, the shift window is that shifting the feature map but not 
            the windows (i.e. the different patches in the same windows). 

            One of the methods is to adding the zero padding into the feature 
            maps and do the feature map rolling. However, it will increase 
            the computing cost. Hence, proposed by the paper, use the rolling 
            method to shift the feature map to-left and to-upper to move half 
            of the window size. (e.g., move 3 patch if window size is 7). 
            """
            shifted_x = tf.roll(input=x,
                                shift=[-self.shift_size, -self.shift_size],
                                axis=[1, 2])
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, shape=(-1, self.window_size *
                                                 self.window_size, channels))

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = tf.reshape(attn_windows, shape=(
            -1, self.window_size, self.window_size, channels))

        shifted_x = window_reverse(attn_windows, self.window_size, height,
                                   width, channels)

        if self.shift_size > 0:
            x = tf.roll(input=shifted_x,
                        shift=[self.shift_size, self.shift_size],
                        axis=[1, 2])
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)

        x = x + x_skip
        x_skip = x

        x = self.layer_normalization_2(x)
        x = self.mlp_block(x)
        x = self.drop_path(x)

        x = x + x_skip

        return x

    def get_config(self):
        config = super(ShiftWindowsTransformerBlock, self).get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_patches": self.num_patches,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "shift_size": self.shift_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PublicSwinIR():
    """ Image Restoration Using Swin Transformer
    https://homes.esat.kuleuven.be/~konijn/publications/2021/Liang5.pdf
    """

    def __init__(self, image_shape, up_factor=1, patch_size=4, num_heads=6,
                 window_size=8, block_num=4, stage_num=6, feature_num=180):
        # SwinIR+: block 6; stage 6; win_size 8; feature 180; head 6
        # SwinIR Lightweight: block 4; stage 6; win_size 8; feature 60, head 6
        width, height, channel = image_shape[0], image_shape[1], image_shape[2]
        self.image_shape = (width // up_factor, height // up_factor, channel)
        self.up_loop = int(up_factor // 2)
        self.patch_size = patch_size
        self.num_heads = num_heads  # 4
        self.window_size = window_size  # 8
        self.shift_size = window_size // 2

        self.block_num = block_num
        self.stage_num = stage_num

        self.drop_prob = 0.03
        self.feature_num = feature_num

    def __call__(self):
        num_patches = (self.image_shape[0] // self.patch_size)**2
        filter_size = (self.patch_size ** 2) * self.image_shape[-1]
        feature_num = self.feature_num
        block = 0
        stage = 0

        # Shallow Feature Extraction
        Inputs = layers.Input(self.image_shape)
        patches = ConvPatchExtractor(
            patch_size=self.patch_size,
            hidden_size=filter_size,
            num_patches=num_patches,
        )(Inputs)

        projection = layers.Dense(units=feature_num)(patches)
        x = tf.keras.layers.LayerNormalization()(projection)
        x_skip = x

        # Residual Swin Transformer Blocks
        num_patch_y = (self.image_shape[1] // self.patch_size)  # 32
        num_patch_x = (self.image_shape[0] // self.patch_size)  # 32
        num_heads = self.num_heads

        for i in range(self.block_num):
            block = i
            block_skip = x

            for j in range(self.stage_num // 2):
                x = ShiftWindowsTransformerBlock(
                    hidden_size=feature_num,
                    num_patches=(num_patch_y, num_patch_x),
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=0,
                    drop_rate=self.drop_prob,
                    block=block,
                    stage=stage,
                )(x)
                stage += 1

                x = ShiftWindowsTransformerBlock(
                    hidden_size=feature_num,
                    num_patches=(num_patch_y, num_patch_x),
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.shift_size,
                    drop_rate=self.drop_prob,
                    block=block,
                    stage=stage,
                )(x)
                stage += 1

            x = tf.reshape(tensor=x, shape=(
                tf.shape(x)[0], num_patch_x, num_patch_y, feature_num))
            x = layers.Conv2D(filters=feature_num, kernel_size=3, strides=1, padding='same')(x)
            x = tf.reshape(tensor=x, shape=(
                tf.shape(x)[0], num_patch_x * num_patch_y, feature_num))

            x = block_skip + x
            stage = 0

        x = x + x_skip
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.reshape(tensor=x, shape=(
            tf.shape(x)[0], num_patch_x, num_patch_y, feature_num))

        # Up-sample process, 2 times, every time up_scale 2
        # Use PixelShuffle method to up-sample (tf.nn.depth_to_space)
        x = layers.Conv2D(filters=16 * feature_num, kernel_size=3, strides=1, padding='same')(x)
        x = tf.nn.depth_to_space(x, 4, 'NHWC')

        x = layers.Conv2D(filters=16 * feature_num, kernel_size=3, strides=1, padding='same')(x)
        x = tf.nn.depth_to_space(x, 4, 'NHWC')

        x = layers.Conv2D(filters=self.image_shape[-1], kernel_size=3, padding='same')(x)
        x = layers.Activation("tanh")(x)

        models = tf.keras.models.Model(inputs=Inputs, outputs=x,
                                       name="SwinIR")
        return models


class ConvPatchExtractor(layers.Layer):
    def __init__(self, patch_size, hidden_size, num_patches):
        super(ConvPatchExtractor, self).__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.conv2d = layers.Conv2D(
            filters=self.hidden_size,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            use_bias=False,
            padding='valid',
        )

    def call(self, inputs, *args, **kwargs):
        x = self.conv2d(inputs)
        B, H, W, C = x.shape
        print(x.shape)
        out = layers.Reshape(target_shape=(H*W, C))(x)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "num_patches": self.num_patches,
        })
        return config