import json

# location of the config file
CONFIGS_PATH = 'configs.json'

def load_configs(path):
    with open(path) as data_file:
        data_loaded = json.load(data_file)
    return data_loaded[0]

configs = load_configs(CONFIGS_PATH)

# location of the img_data files
IMGS_DIR = configs["imgs dir"]
# location of the tfrecords files
TRAINING_PATH = configs["training path"]
TESTING_PATH = configs["testing path"]
VALIDATION_PATH = configs["validation path"]

# image size SIZExSIZE
IMG_SIZE = configs["image resize"]
# testing percentage
TRAIN_PERCENTAGE = configs["train percentage"]

# optimization variables
LEARNING_RATE = configs["learning rate"]
DECAY_RATE = configs["decay rate"]

EPOCHS = configs["max epochs"]
BATCH_SIZE = configs["batch size"]

CHECKPOINT_INTERVAL = configs["checkpoint interval"]
CHECKPOINT_DIR = configs["checkpoint dir"]

print("Configs loaded.")
