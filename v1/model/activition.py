import torch
from torch.autograd import Function
from torch.nn import Module

def fly(x,d=0.5):
    if -d<=x<=d:
        return x
    elif x>d:
        return d+1-d/x
    else:
        return -d-1-d/x

def demo_fly():
    i=-5.0
    while i<5.0:
        print(i,fly(i),fly(i)/i)
        i+=0.1
    i=2
    while i<2**10:
        print(i,fly(i),fly(i)/i)
        i*=2

def penalized_tanh(x):
    alpha = 0.25
    return torch.max(torch.tanh(x), alpha*torch.tanh(x))

def relu(x):
    return (x>0)*x

class Relu0(Function):
    @staticmethod
    def forward(ctx, input,d=1):
        ctx.save_for_backward(input)  # save input for backward pass
        x=torch.clone(input)
        x[x<0]=0
        return x

    @staticmethod
    def backward(ctx, grad_output,d=1):
        input, = ctx.saved_tensors # restore input from context
        grad_output[input<0]= 0
        return grad_output

relu0=Relu0.apply

class Wing0(Function):
    @staticmethod
    def forward(ctx, input,d=1):
        ctx.save_for_backward(input)  # save input for backward pass
        # x=torch.clone(input)
        # recip=d/torch.reciprocal(input)
        recip=d/input
        input[input<-d]= (-d-1-recip)[input<-d]
        input[input>d]=(d+1-recip)[input>d]
        return input


    @staticmethod
    def backward(ctx, grad_output,d=1):
        input, = ctx.saved_tensors # restore input from context
        recip2=d/torch.pow(input,2)
        grad_output[input>d]= recip2[input>d]
        grad_output[input<-d]= recip2[input<-d]
        return grad_output

class Wing1(Function):
    @staticmethod
    def forward(ctx, input,d=1):
        ctx.save_for_backward(input)  # save input for backward pass
        x=torch.clone(input)
        # recip=d/torch.reciprocal(input)
        recip=-d/x
        x[x<-d]= (-d-1+recip)[x<-d]
        x[x>d]=(d+1+recip)[x>d]
        return x


    @staticmethod
    def backward(ctx, grad_output,d=1):
        input, = ctx.saved_tensors # restore input from context
        recip2=d/torch.pow(input,2)
        grad_output[input>d]= recip2[input>d]
        grad_output[input<-d]= recip2[input<-d]
        return grad_output

# oom
def wing(x,d=1):
    # return (x>d)*(d+1-1/x)+(x<-d)*(-d-1-1/x)+x.clamp(min=-d,max=d)
    # return torch.tanh(x)+1-(x>d)/x
    # return torch.tanh(x)+torch.log(x.clamp(min=d))
    # return torch.tanh(x)+(1-1/x.clamp(min=d))
    return torch.sign(x)*torch.min(torch.abs(x),1-d/torch.abs(x))

def ptanh(x):  # penalized tanh
    return torch.tanh(x)-(x<0)*torch.tanh(x)*0.75
    # return torch.tanh(x)+(x>0)*torch.tanh(x)*3

class LinearNorm(Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self,config=None,eps=None):
        super().__init__()

    def forward(self, x):
        m = x.mean(-1, keepdim=True)
        return x-m

class LinearNorm(Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self,config=None,eps=None):
        super().__init__()

    def forward(self, x):
        m = x.mean(-1, keepdim=True)
        return x-m

class SqrtNorm(Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self,config=None,eps=None):
        super().__init__()

    def forward(self, x):
        m=torch.sign(x)*torch.sqrt(torch.abs(x))
        # m = x.mean(-1, keepdim=True)
        return m

class WingNorm(Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self,config=None,eps=None):
        super().__init__()
        self.act=wing
        self.norm=SqrtNorm()
        self.norm2=torch.nn.LayerNorm(512,1e-12)
    def forward(self, x):
        w=self.act(x)
        m=self.norm(w-w.mean(-1, keepdim=True))
        n=self.norm2(w)
        return n


def awing(input,grad_output,d=1):
    recip2 = d / torch.pow(input, 2)
    grad_output[input > d] = recip2[input > d]
    grad_output[input < -d] = recip2[input < -d]
    return grad_output

def wing_test():
    seq=[-3,-0.3,0,0.3,3]
    print(  [  fly(x) for x in seq ] )
    x=torch.rand((70,128,512))
    p=wing(x)
    print(f" wing {p}")
    g=awing(x,p)
    print(f" awing {g}")
    norm=WingNorm()
    n=norm(x)
    print(f"norm {n}")

if __name__=="__main__":
    demo_fly()
    wing_test()
