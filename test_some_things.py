import torch

import torch.nn as nn

# bilstm = nn.LSTM(256, 128, bidirectional=True)
# features_bilstm1 = torch.randn(64, 256)
# features_bilstm2 = torch.randn(64, 256)
# features_bilstm = torch.stack([features_bilstm1, features_bilstm2],0)
# print(features_bilstm.shape)
# features_bilstm, (_, _) = bilstm(features_bilstm)

input = torch.randn(64, 2048, 1, 1)
conv = nn.Conv1d(2048, 256, kernel_size=1)
output = conv(input)
print(output.shape)
