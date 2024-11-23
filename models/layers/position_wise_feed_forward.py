from torch import nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_out=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.linear2(x)

        return x