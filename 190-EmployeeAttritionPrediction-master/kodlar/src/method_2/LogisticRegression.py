import torch


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.batch_norm = torch.nn.BatchNorm1d(input_size)
        self.drop_out = torch.nn.Dropout(0.2)

    def forward(self, x):
        output = self.linear(x)
        output = self.batch_norm(output)
        output = self.drop_out(output)
        output = torch.relu(output)
        return output
