import numpy as np
import os
import cv2

conv1_weight = np.array([[[[-1, -1, 1],
                           [-1, 0, 1],
                           [-1, 1, 1]],

                          [[-1, -1, -1],
                           [-1, 0, 1],
                           [1, 1, 1]]]])

conv2_weight = np.array([[[[-1, -1, 1],
                           [-1, 2, -1],
                           [1, 1, 1]],

                          [[1, -1, 0],
                           [-1, -1, -1],
                           [1, -1, 1]]]])


class model():
    def __init__(self) -> None:
        self.conv_weight = conv1_weight
        self.conv_weight2 = [conv1_weight, conv2_weight]

    def forward(self, x) -> np.ndarray:
        x = self.convolution(x, kernel_size=3, in_channel=1,  # Conv1
                             out_channel=2, padding=1, kernel_weight=self.conv_weight)

        x = self.maxpool(x, kernel_size=2)  # Maxpool1

        x = self.convolution(x, kernel_size=3, in_channel=2,  # Conv2
                             out_channel=2, padding=1, kernel_weight=self.conv_weight2)

        x = self.maxpool(x, kernel_size=2)  # Maxpool2

        x = np.reshape(x, (x.shape[0], -1))  # Flatten

        x = np.append(x, np.expand_dims(
            np.array([1] * x.shape[0]), axis=1), axis=1)
        return x

    def convolution(self, input_image, kernel_size, in_channel, out_channel, padding, kernel_weight) -> np.ndarray:

        convolution_kernel = kernel_weight
        for batch in range(input_image.shape[0] - 1):  # Prepare Filter Size
            convolution_kernel = np.concatenate(
                (convolution_kernel, kernel_weight), axis=0)

        feature_map = np.zeros(
            (tuple([input_image.shape[0]]) + tuple([out_channel]) + input_image.shape[2:]))

        input_image = np.pad(input_image, ((0, 0), (0, 0), (padding, padding),
                             (padding, padding)), mode='constant', constant_values=(0, 0))  # Paddind image

        if (in_channel == 1):  # For First Layer
            for w in range(feature_map.shape[2]):
                for h in range(feature_map.shape[3]):
                    feature_map[:, :, w, h] = np.sum(np.sum(
                        input_image[:, :, w:w+kernel_size, h:h+kernel_size] * convolution_kernel, axis=2), axis=2)
            return feature_map
        else:  # For in_channel = 2
            for channel in range(out_channel):  # Channelwise convolution
                for w in range(feature_map.shape[2]):
                    for h in range(feature_map.shape[3]):
                        feature_map[:, channel, w, h] = np.sum(np.sum(np.sum(
                            input_image[:, :, w:w+kernel_size, h:h+kernel_size] * convolution_kernel[channel], axis=2), axis=2), axis=1)

            return feature_map

    def maxpool(self, input_image, kernel_size, stride=2) -> np.ndarray:

        feature_map = np.zeros(  # Set feature map size
            (input_image.shape[:2] + tuple(int(shape / kernel_size) for shape in input_image.shape[2:])))

        for w in range(feature_map.shape[2]):
            for h in range(feature_map.shape[3]):
                feature_map[:, :, w, h] = np.max(np.max(  # Calculate maxpool
                    input_image[:, :, w*stride:w*stride+kernel_size, h*stride:h*stride+kernel_size], axis=2), axis=2)

        return feature_map
