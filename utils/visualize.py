import matplotlib.pyplot as plt
import numpy as np


def plot_image(img, true=None, pred=None):
    img = np.squeeze(img)

    n = 1
    if not true is None:
        true = np.squeeze(true)
        n += 1
    if not pred is None:
        pred = np.squeeze(pred)
        n += 1

    i = 1
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, n, i)
    ax.text(0, -3, 'Image', fontsize=10)
    ax.imshow(img, cmap='gray')
    ax.axis('off')

    if not true is None:
        i += 1
        ax = fig.add_subplot(1, n, i)
        ax.text(0, -3, 'Target', fontsize=10)
        ax.imshow(true, cmap='jet')
        ax.axis('off')

    if not pred is None:
        i += 1
        ax = fig.add_subplot(1, n, i)
        ax.text(0, -3, 'Prediction', fontsize=10)
        ax.imshow(pred, cmap='jet')
        ax.axis('off')


def plot_dataset(dataset):
    for data in dataset:
        imgs, maps = data

        for img, true in zip(imgs, maps):
            plot_image(img, true=true)

        plt.show()
        break


def plot_images(x_data, y_true, y_pred):
    for img, true, pred in zip(x_data, y_true, y_pred):
        plot_image(img, true=true, pred=pred)

    plt.show()


def plot_predictions(x_data, y_pred):
    for img, pred in zip(x_data, y_pred):
        plot_image(img, pred=pred)

    plt.show()
