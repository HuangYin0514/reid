# from models.bilstm import FullBiLSTM as inception

# input_dim=512
# hidden_dim=512
# vocab_size=128
# batch_first=True
# freeze =True
# model = inception(
#     input_dim,
#     hidden_dim,
#     vocab_size,
#     batch_first,
#     dropout=0.7,
#     freeze=freeze,
# )
import torch
import torch.nn as nn

x = torch.rand(6, 20, 256)
net = nn.LSTM(256, 256, bidirectional=True)

output, (h0, c0) = net(x)
print(output.shape)


