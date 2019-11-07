import tensorflow as tf
import json


def from_json(model_name, epoch):
    # Load model
    with open(f'./results/{model_name}/model.json', 'r') as f:
        json_config = json.load(f)
    model = tf.keras.models.model_from_json(json_config)

    # Recover weights
    model.load_weights(f'./results/{model_name}/epoch-{epoch:03d}.ckpt')
    return model
