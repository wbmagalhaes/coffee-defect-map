import matplotlib.pyplot as plt
import numpy as np


def plot_dataset(dataset):
    for data in dataset:
        imgs, maps = data

        fig = plt.figure(figsize=(8, 8))

        x = np.squeeze(imgs[0])
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(x, cmap='gray')
        ax.axis('off')

        y = np.squeeze(maps[0])
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(y, cmap='jet')
        ax.axis('off')

        plt.show()
        break
