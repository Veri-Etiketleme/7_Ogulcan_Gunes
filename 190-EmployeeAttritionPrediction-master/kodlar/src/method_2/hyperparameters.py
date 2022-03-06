import torch
from LogisticRegression import LogisticRegression


batch_size = 256
output_size = 2
n_iter = 10000
print_freq = n_iter // 10
criterion = torch.nn.CrossEntropyLoss()
learning_rate = 0.005
features_not_to_use = ["EmployeeCount", "Over18", "StandardHours"]


def get_model(_device, _input_size):
    global model, optimizer, scheduler, input_size, device
    input_size = _input_size
    device = _device
    model = LogisticRegression(input_size, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, n_iter // 3)
