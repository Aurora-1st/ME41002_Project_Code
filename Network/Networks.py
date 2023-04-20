from Network import Layers

"""
Models:
srcnn
fsrcnn 
vdsr
"""


def srcnn(img_shape):
    return Layers.SRCNN(img_shape=img_shape)


def fsrcnn(img_shape):
    return Layers.FSRCNN(img_shape=img_shape)


def vdsr(img_shape):
    return Layers.VDSR(img_shape=img_shape)




