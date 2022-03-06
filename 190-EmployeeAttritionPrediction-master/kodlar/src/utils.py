import random
import datetime
import torch

model = None
data = None


def now_to_string(format):
    now = datetime.datetime.now()
    string = now.strftime(format)
    return string


def elapsed_from(now):
    then = datetime.datetime.now()
    took = then - now
    mm, ss = divmod(took.seconds, 60)
    hh, mm = divmod(mm, 60)
    return "%dh %dm %ds" % (hh, mm, ss)


def get_batch(ind, to_use):
    batch = {"input": [], "label": []}

    for i in ind:
        batch["input"].append(data[to_use]["input"][i])
        batch["label"].append(data[to_use]["label"][i])

    batch["input"] = torch.Tensor(batch["input"]).to(model.device)
    batch["label"] = torch.LongTensor(batch["label"]).to(model.device)

    return batch


def get_random_train_batch(train_input_ind):
    rand_ind = random.sample(train_input_ind, model.batch_size)
    return get_batch(rand_ind, "train")


def train_batch(batch):
    model.optimizer.zero_grad()
    output = model.model(batch["input"])
    loss = model.criterion(output, batch["label"])
    loss.backward()
    model.optimizer.step()

    if model.scheduler is not None:
        model.scheduler.step()

    return loss.item() / model.batch_size
