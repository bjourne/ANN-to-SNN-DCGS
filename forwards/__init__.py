from .encoder import *
from .decoder import *
import types
from utils import *
import torch
import copy
import numpy as np


def forward_snn_diff_leaky_rate_s(self, x):
    output = []
    output.append(self.init_forward(torch.zeros_like(x)))
    output.append(self.init_forward(x))
    mul = 1
    for i in range(1, self.T):
        mul /= self.tau
        output.append(self.init_forward(torch.zeros_like(x))*mul)
    return decodeoutput(torch.stack(output, dim=0))

def forward_snn_diff_leaky_rate_m(self, x):
    x = add_dimention_diff(x, self.T)
    x = self.merge(x)
    out = self.init_forward(x)
    out = self.expand(out)
    mul = 1
    for i in range(2,self.T+1):
        mul /= self.tau
        out[i]*=mul
    return decodeoutput(out)

def forward_snn_diff_rate_m(self, x):
    x = add_dimention_diff(x, self.T)
    x = self.merge(x)
    out = self.init_forward(x)
    out = self.expand(out)
    return decodeoutput(out)

def forward_snn_leaky_rate_s(self, x):
    output = []
    mul = 1
    for i in range(self.T):
        output.append(self.init_forward(x/mul)*mul)
        mul /= self.tau
    return torch.stack(output, dim=0)

def forward_snn_leaky_rate_m(self, x):
    x = add_dimention(x, self.T)
    for i in range(1,self.T):
        x[i]=x[i-1]/self.tau
    x = self.merge(x)
    out = self.init_forward(x)
    out = self.expand(out)
    mul = 1
    for i in range(self.T):
        out[i] *= mul
        mul /= self.tau
    return out


def add_dimention(x, T):
    x.unsqueeze_(0)
    x = x.repeat(T, 1, 1, 1, 1)
    return x

def add_dimention_diff(x, T):
    x.unsqueeze_(0)
    x = x.repeat(T+1, 1, 1, 1, 1)
    x[0] = 0
    x[2:] = 0
    return x



def forward_snn_rate_m2(self, x):
    x = add_dimention(x, self.T)
    x = self.merge(x)
    out = self.init_forward(x)
    return out

def forward_snn_rate_m3(self, x):
    out = self.init_forward(x)
    out = self.expand(out)
    return out

def forward_snn_diff_rate_m2(self, x):
    x = add_dimention_diff(x, self.T)
    x = self.merge(x)
    out = self.init_forward(x)
    return out

def forward_snn_diff_rate_m3(self, x):
    out = self.init_forward(x)
    out = self.expand(out)
    return {key:decodeoutput(out[key]) for key in out}
