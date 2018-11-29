import json
from utils import config

def load_labelmap(path):
    with open(path) as data_file:
        data_loaded = json.load(data_file)

    labels = []
    for label in data_loaded:
        label_dict = {
                'id': int(label['id']),
                'name': str(label['name']),
                'weight': float(label['weight'])
                }
        labels.append(label_dict)

    print("Labels loaded.")
    return labels

def index_of_label(name):
    for label in labels:
        if label['name'] == name:
            return label['id']

    return -1

def weight_of_label(name):
    for label in labels:
        if label['name'] == name:
            return label['weight']

    return 0

def name_of_idx(idx):
    for label in labels:
        if label['id'] == idx:
            return label['name']

    return "none"

def weight_of_idx(idx):
    for label in labels:
        if label['id'] == idx:
            return label['weight']

    return 0

labels = load_labelmap('label_map.json')
count = len(labels)
