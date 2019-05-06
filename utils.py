from torch.autograd import Variable
import torch

def convert_long_tensor(var,use_cuda):
    var=torch.LongTensor(var)
    if use_cuda:
        var=var.cuda(async=True)
    return var

def convert_long_variable(var,use_cuda):
    return Variable(convert_long_tensor(var,use_cuda))