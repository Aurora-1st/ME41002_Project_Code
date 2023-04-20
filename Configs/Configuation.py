import tensorflow as tf
from Train import LossFunc


class Param:
    def __init__(self):
        # TODO: parameters for keras.fit function
        self.batch_size = 4
        self.epochs = 1000

        # TODO: parameters for keras.compile function
        self.loss = [LossFunc.vgg19_loss]
        self.learning_rate = 1e-5
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # TODO: set the name of the saved history
        self.model_name = 'DenseSTSR_v3'
        self.times = 1

        # TODO: number of train/validation datasets
        self.HR_train_path = 'Dataset/train_max_5'
        self.LR_train_path = 'Dataset/train_max_5'
        self.HR_valid_path = 'Dataset/valid_max_5'
        self.LR_valid_path = 'Dataset/valid_max_5'
        self.HR_test_path = 'Dataset/test_max_5'  # ground truth
        self.LR_test_path = 'Dataset/test_max_5'  # input
        self.num_train_data = 2220
        self.num_valid_data = 300
        self.num_test_data = 60

        # TODO: shape of the datasets and labels
        self.width = 96  # LR
        self.height = 96  # LR
        self.channel = 3
        self.img_format = 'jpg'
        self.norm = 255  # uint8 png and jpeg: 255; uint16 png: 65535
        self.img_shape = (self.width, self.height, self.channel)
        self.img_size = (self.width, self.height)

        # TODO: Transformer params
        # self.patch_size = 16
        # self.embed_dim = 768
        self.depth = 12
        # self.num_heads = 12
        self.has_logits = True