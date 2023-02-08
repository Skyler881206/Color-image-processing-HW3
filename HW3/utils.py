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

    def forward(self, x, flatten=True) -> np.ndarray:
        x = self.convolution(x, kernel_size=3, in_channel=1,  # Conv1
                             out_channel=2, padding=1, kernel_weight=self.conv_weight)

        x = self.maxpool(x, kernel_size=2)  # Maxpool1

        x = self.convolution(x, kernel_size=3, in_channel=2,  # Conv2
                             out_channel=2, padding=1, kernel_weight=self.conv_weight2)

        x = self.maxpool(x, kernel_size=2)  # Maxpool2

        if (flatten == True):
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


class train():
    def __init__(self, preditcion, ground_truth, learning_rate, epoch) -> None:
        self.weight = np.zeros((9, 1))
        self.prediction = preditcion
        self.ground_truth = ground_truth
        self.learning_rate = learning_rate
        self.epoch = epoch

    def loss_function(self, prediciton: np.ndarray, ground_truth: np.ndarray) -> None:
        # MSE
        return np.mean((ground_truth - prediciton * self.weight) ** 2)

    def backpropagation(self):
        loss = self.loss_function(self.prediction, self.ground_truth)
        gradient = np.gradient(loss, self.weight)

        return

    def start_training(self):
        return


if __name__ == "__main__":
    import config
    from natsort import natsorted

    BATCH = config.BATCH

    images = []
    labels = []

    for root, dirs, files in os.walk(config.IMAGE_PATH):
        files = natsorted(files)  # Sort by file name in the list
        for file in files:
            if (".png" in file):
                images.append(os.path.join(root, file))
                continue

            if (".txt" in file):
                file = open(os.path.join(root, file))
                while True:
                    line = file.readline()
                    if not line:
                        break
                    labels.append(int(line.strip()))
                file.close()

    for index in range(0, len(images), BATCH):
        model_arch = model()  # Read model
        label_list = []
        image_list = []

        for idx in range(BATCH):  # Prepare Data
            image = cv2.imread(images[index + idx], cv2.IMREAD_GRAYSCALE)
            image_list.append(image)  # Save original image
            # Resize image to 8x8
            image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC)
            # Expand dim to fit model
            image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
            # Prepare label list
            label = np.expand_dims(
                np.mod(np.array(labels[index + idx]), 2), axis=0)

            label_list.append(labels[index + idx])
            if idx == 0:  # First Iteration
                input_image = image
                input_label = label

            else:
                input_image = np.concatenate(
                    (input_image, image), axis=0)  # Concate image for batch
                input_label = np.concatenate((input_label, label), axis=0)
        label_list = list(set(label_list))

        prediction = model_arch.forward(input_image, False)  # Get feature map

    pass
