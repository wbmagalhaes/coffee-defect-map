import tensorflow as tf

from CoffeeUNet import create_model

weights_path = './results/coffeeunet18.h5'
out_path = './results/coffeeunet18_v0.1.tflite'

model = create_model()
model.load_weights(weights_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

open(out_path, 'wb').write(tflite_model)
