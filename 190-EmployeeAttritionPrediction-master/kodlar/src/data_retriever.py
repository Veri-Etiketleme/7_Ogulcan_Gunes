import sys
import pandas as pd
from random import shuffle
from codes import codes
import numpy as np


def preprocess_data(data_path, features_not_to_use):
    original_data = pd.read_csv(data_path)

    data = {
        "input": [],
        "label": []
    }

    for i in range(len(original_data)):
        data["input"].append([])
        for col in original_data.columns:
            if col in features_not_to_use:
                continue
            put = data["input"][-1]
            value = original_data[col][i]
            if isinstance(value, str):
                value = codes[col][value]
                if col == "Attrition":
                    put = data["label"]
            put.append(value)

    return data


def split_data(data):
    indices = [i for i in range(len(data["input"]))]
    shuffle(indices)

    split_1 = (len(data["input"]) // 10) * 7
    split_2 = split_1 + (len(data["input"]) - split_1) // 2

    s_data = {
        "train": {"input": data["input"][:split_1], "label": data["label"][:split_1]},
        "valid": {"input": data["input"][split_1 + 1:split_2], "label": data["label"][split_1 + 1:split_2]},
        "test": {"input": data["input"][split_2 + 1:], "label": data["label"][split_2 + 1:]},
    }

    return s_data


def analyze_data(data):
    sys.stdout.write("There are %d columns.\n" % len(data.columns))
    types = {}

    for col in data.columns:
        unique = set()
        for d in data[col]:
            unique.add(d)
            if isinstance(d, str):
                types[col] = "str"
            else:
                types[col] = "int"

        if types[col] == "str":
            sys.stdout.write(col + '\n')
            sys.stdout.write("%d unique values in column %s which are:\n" % (len(unique), col))
            for u in unique:
                sys.stdout.write("%s " % u)
            sys.stdout.write('\n')


def z_score_normalize(input_data):
    mean = np.mean(input_data, axis=0)
    std = np.std(input_data, axis=0)
    z_score = (input_data - mean) / std

    return z_score
