import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 128
max_len = 256
d_model = 512
n_layer = 6
n_head = 8
ffn_hidden = 2048
drop_prob = 0.1

init_lr = 1e-5
factor = 0.9
patience = 10
warmup = 100
epoch = 10000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
