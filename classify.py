import numpy as np

from CoffeeUNet import create_model

from utils import data_reader, visualize

sample_paths = [
    'C:/Users/Usuario/Desktop/cafe_imgs/amostras/84A',
    'C:/Users/Usuario/Desktop/cafe_imgs/amostras/248A'
]

x_data = data_reader.load_images(sample_paths, final_size=128)
x_data = np.array(x_data).astype(np.float32) / 255.

model = create_model()
model.load_weights('./results/coffeeunet18.h5')

y_pred = model.predict(x_data)

visualize.plot_predictions(x_data[:4], y_pred[:4])
