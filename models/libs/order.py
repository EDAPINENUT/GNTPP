import torch
from torch.autograd import Variable
from torch import nn
class MaskBatch():
    "object for holding a batch of data with mask during training"
    def __init__(self, pad_index):
        self.pad = pad_index

    def make_std_mask(self, tgt):
        "create a mask to hide padding and future input"
        tgt_mask = (tgt != self.pad).unsqueeze(-2).to(tgt)
        tgt_mask = tgt_mask & Variable(self.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(tgt)
        return tgt_mask
    
    def subsequent_mask(self, size):
        "mask out subsequent positions"
        atten_shape = (1,size,size)
        mask = torch.triu(torch.ones(atten_shape), diagonal=1)
        m = mask == 0
        return m

class TimeDecayer(nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        self.alphas = nn.Parameter(torch.randn((heads,), requires_grad = True))
        
    def forward(self, lag_matrix):
        return (- self.alphas[None,:,None,None].square() * lag_matrix[:,None,:,:]).exp()