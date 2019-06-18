import tensorflow as tf

model_id = 'CoffeeUNet18'
model_dir = 'saved_models/' + model_id

input_arrays = ["inputs/img_input"]
output_arrays = ["result/dmap", 'result/maxima']
input_shapes = {
    "inputs/img_input": [1, 256, 256, 1]
}

converter = tf.lite.TFLiteConverter.from_saved_model(
    model_dir,
    input_arrays=input_arrays,
    input_shapes=input_shapes,
    output_arrays=output_arrays,
    signature_key="predict"
)

# converter.allow_custom_ops = True

tflite_model = converter.convert()

print(converter._input_tensors)
print(converter._output_tensors)

open("coffeeunet18_v1_1.0.tflite", "wb").write(tflite_model)
