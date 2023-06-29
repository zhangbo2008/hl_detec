import torch
import torch.nn as nn

prediction=torch.ones(2,1,5) # (seq_len,batch_size,output_size)
b_y=torch.randn(2,1,5)# (seq_len,batch_size,output_size)

b_y = b_y.type(torch.FloatTensor).to('cpu')
prediction = prediction.type(torch.FloatTensor).to('cpu')
loss_func = nn.MSELoss()
loss = loss_func(prediction, b_y)
a=torch.mean((prediction-b_y)**2)
print('a====loss')





