import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)


    def forward(self, x):
        output = torch.relu(self.linear(x))
        return output
