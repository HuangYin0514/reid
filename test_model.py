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


x=torch.rand(2,5,3)    # 序列长度为5，输入尺度为3
net=nn.LSTM(3,4,6,bidirectional=False,batch_first=True)

output, (h0,c0)= net(x)
print(output)