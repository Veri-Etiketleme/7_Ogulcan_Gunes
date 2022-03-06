import importlib
import utils
from utils import *
import sys
import data_retriever as dr
import pickle
from os import path

model = None
data = None


def init(device, data_path, method):
    global model, data
    model = importlib.import_module(method + ".hyperparameters")
    utils.model = model
    save_path = "../data/%s-data.pkl" % method

    if path.exists(save_path):
        data = pickle.load(open(save_path, "rb"))
    else:
        data = dr.preprocess_data(data_path, model.features_not_to_use)
        if method != "method_1":
            data["input"] = dr.z_score_normalize(data["input"])
        data = dr.split_data(data)
        pickle.dump(data, open(save_path, "wb"))

    utils.data = data
    model.device = device
    model.get_model(device, len(data["train"]["input"][0]))

    return data


def train():
    train_input_ind = [i for i in range(len(data["train"]["input"]))]

    sys.stdout.write("------------------------------------------\n")
    sys.stdout.write(" Training started at %s." % now_to_string("%F %T") + '\n')
    sys.stdout.write("------------------------------------------\n")

    now = datetime.datetime.now()
    model.model.train()
    avg_loss = 0
    for iter in range(1, model.n_iter + 1):
        batch = get_random_train_batch(train_input_ind)
        loss = train_batch(batch)
        avg_loss += loss

        if iter % model.print_freq == 0:
            sys.stdout.write("%3g%%: %.6f " % ((iter / model.n_iter) * 100, avg_loss / model.print_freq))
            sys.stdout.write("(" + elapsed_from(now) + ')\n')
            avg_loss = 0
    sys.stdout.write('\n')


def evaluate(to_use):
    sys.stdout.write("--------------------------------------------\n")
    sys.stdout.write(" Validation started at %s." % now_to_string("%F %T") + '\n')
    sys.stdout.write("--------------------------------------------\n")

    now = datetime.datetime.now()
    model.model.eval()
    correct = 0
    iter = 0

    while iter < len(data[to_use]["input"]):
        try:
            assert iter + model.batch_size < len(data[to_use]["input"])
            ind = [i for i in range(iter, iter + model.batch_size)]
        except AssertionError:
            ind = [i for i in range(iter, len(data[to_use]["input"]))]

        batch = get_batch(ind, to_use)
        output = model.model(batch["input"])

        for b in range(len(ind)):
            if torch.argmax(output[b]) == batch["label"][b]:
                correct += 1

        iter += model.batch_size

    sys.stdout.write("Elapsed " + elapsed_from(now) + '\n')
    return correct / len(data[to_use]["input"])
