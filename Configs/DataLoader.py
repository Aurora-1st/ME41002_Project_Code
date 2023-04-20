import pathlib
import tensorflow as tf


def zip_and_batch_ds(inputs_x, inputs_y, batch_size):

    ds = tf.data.Dataset.from_tensor_slices((inputs_x, inputs_y))
    return ds.batch(batch_size)


class DataLoader:
    def __init__(self, img_size, dim, img_batches, file_path=None):

        self.dim = dim
        self.img_size = img_size
        self.img_batches = img_batches

        self.data_path = pathlib.Path(file_path)
        self.image_path = list(self.data_path.glob('*.jpg'))
        self.all_image_paths = [str(path) for path in self.image_path]
        self.image_count = len(self.all_image_paths)

        self.path_ds = tf.data.Dataset.from_tensor_slices(self.all_image_paths)
        self.image_ds = self.path_ds.map(self.load_and_preprocess_image)

    def preprocess_image(self, image):
        image = tf.image.decode_jpeg(contents=image, channels=self.dim)

        image = tf.image.resize(images=image,
                                size=self.img_size,
                                method='bicubic')
        image = image / 255.
        return image

    def load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self.preprocess_image(image)

    def return_tensor(self) -> object:
        ds = self.image_ds
        ds = ds.repeat()
        ds = ds.batch(self.img_batches)
        image_batch = next(iter(ds))
        return image_batch
