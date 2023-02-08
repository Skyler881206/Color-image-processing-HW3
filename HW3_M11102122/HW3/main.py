import config
from natsort import natsorted
import os
import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

if __name__ == "__main__":

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
        model = utils.model()  # Read model
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

        prediction = model.forward(input_image)  # Get feature map

        # Transpose shape to (9, BATCH) for linear regression
        trans_prediction = np.transpose(prediction, (1, 0))
        # Prepare for linear regression
        gt_label = np.expand_dims(np.array(input_label), axis=1)

        linear_gression_function = np.dot(np.linalg.inv(  # Linear regression
            np.dot(trans_prediction, prediction)), np.dot(trans_prediction, gt_label))

        confusion_matrix = np.zeros((2, 2))  # Set confusion matrix size
        prediction_list = []
        for idx in range(BATCH):
            prediction_number = np.where(  # Check prediction and gt is equal or not
                np.sum(np.squeeze(prediction[idx]) * np.squeeze(linear_gression_function)) > 0.5, 1, 0)
            prediction_list.append(prediction_number)  # Save prediction value
            confusion_matrix[int(np.mod(gt_label[idx][0], 2)),  # Confusion matrix value plus 1
                             int(np.mod(prediction_number, 2))] += 1

        print("Confusion Matrix {label}\n{fm}\n".format(
            label=label_list, fm=confusion_matrix))

        random_list = random.sample(range(200), 15)  # Randon extract data
        # Set figure size for 15 figures and confusion matrix
        fig, ax = plt.subplots(3, 6)

        fig.suptitle("COMPARE {LABEL}".format(LABEL=label_list))

        for row in range(3):
            for column in range(5):
                idx = 5 * row + column
                color = "blue" if prediction_list[random_list[idx]  # If prediciont == gt label: set blue color, o.w. set red color
                                                  ] == gt_label[random_list[idx]][0] else "red"
                ax[row, column].imshow(  # Set image for grayscale
                    np.squeeze(image_list[random_list[idx]]), cmap="gray")

                ax[row, column].set_title("{label}({idx})".format(label=label_list[0] + input_label[random_list[idx]],  # Set title for image number and index
                                                                  idx=index + random_list[idx] + 1), color=color)

                ax[row, column].axis("off")

        ax[0, 5].axis("off")  # Close axis for right hand side
        ax[1, 5].axis("off")

        ax[2, 5].matshow(confusion_matrix)

        ax[2, 5].set_xticks(  # Set confusioon matrix label
            np.arange(len(label_list)), labels=label_list)
        ax[2, 5].set_yticks(
            np.arange(len(label_list)), labels=label_list)

        for i in range(len(label_list)):
            # Set confusion matrix with value of count for prediction
            for j in range(len(label_list)):
                ax[2, 5].text(i, j, str(int(confusion_matrix[i, j])),
                              va="center", ha="center")

        fig.tight_layout() # Let figure not too tight

        plt.savefig(os.path.join(config.SAVE_PATH, # Save result figure
                    str(int(index/BATCH)) + ".jpg"))
        plt.clf()
