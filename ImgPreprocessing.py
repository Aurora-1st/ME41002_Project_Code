import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom


class EnFaceProjection:
    """
    extract the 2D images from 3D dicom file via Z axis

    :param file_path: the file path of the input dicom image
    :param depth: the sampling range of Z axis, from depth[0] to depth[1]
    :param step: make 'step' number of pixels to one point
    """
    def __init__(self, file_path, depth=None, step=10):
        if depth is None:
            depth = [40, 130]

        self.dicom = pydicom.read_file(file_path)
        print(np.shape(self.dicom))
        # shape of image array [x, y, z]
        self.img = np.array(self.dicom.pixel_array)
        print(np.shape(self.img))
        self.img = np.transpose(self.img, [0, 2, 1])
        print(np.shape(self.img))
        self.depth = depth
        self.step = step

    def call(self, f_name=None):
        step = self.step
        proj_s = self.depth[0]  # start of sampling point
        proj_e = self.depth[1]  # end of sampling point
        norm_factor = 225.
        proj_list = []
        sp = 0

        for _ in range(int((proj_e-proj_s)/step)):
            temp_proj = self.img[:, :, (sp + proj_s):(sp + proj_s + step)]
            assert temp_proj.size != 0
            # max, sum, mean
            temp_proj = np.max(temp_proj, axis=-1)
            temp_proj = temp_proj / step
            temp_proj = temp_proj / norm_factor
            proj_list.append(temp_proj)
            sp += step  # step = 10
            print(sp)
            self.plot_enface(proj_list, sp, f_name)

    def plot_enface(self, proj_list=None, f_number=None, f_name=None):
        print('enface')
        img = np.max(proj_list, axis=0)
        img = np.squeeze(img)
        plt.imshow(img, 'gray')
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        plt.imsave('Dataset/OralOCT_max_5/enface_' + f_name + str(f_number) + '.jpg', img, cmap='gray')


if __name__ == '__main__':
    folder_path = r"Dataset\OralOCT_dicom_flow"
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        print(file_path)

        EnFaceProjection(file_path, depth=[100, 250], step=5).call(file_name)



