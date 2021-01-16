import math
import numpy as np
import matplotlib.pyplot as plt

class PoorLR(object):
    def __init__(self,optimizer,base_lr,total_stage,warm_up=0.1):
        self.base_lr=base_lr
        self.optimizer = optimizer
        self.warm_up = warm_up
        self.total_stage = total_stage-1
        self.now_stage = -1
        self.total_stair=1
        self.now_stair=0
        # self._lr = 0
    
    def get_progress(self):
        p1=self.now_stage/self.total_stage
        p2=self.now_stair/self.total_stair
        p0=1/self.total_stage
        p=(1-p0)*p1+p0*p2        
        return p,p1,p2

    def feed(self,total_stair):
        self.total_stair=total_stair
        self.now_stage+=1
        self.now_stair=1        

    def get_lr(self):
        p,p1,p2=self.get_progress()
        # p0=1/self.total_stage
        # p=(1-p0)*p1+p0*p2
        # wave=math.sin(math.pi*p2)
        if p<=self.warm_up:
            lr0= p/self.warm_up
        else:
            lr0=(1-p)/(1-self.warm_up)

        def polish(r):
            eps=1e-7
            r=max(eps,r)
            r=min(1-eps,r)
            return r
        lr0=polish(lr0)
        wave=math.sin(math.pi*p2)
        lr=lr0*(0.5+wave*0.5)

        return lr*self.base_lr
        # return lr

    def step(self):
        self.now_stair+=1
        lr = self.get_lr()
        if self.optimizer:
            for p in self.optimizer.param_groups:
                p['lr'] = lr


total_stage=20
lines=100

scheduler=PoorLR(None,0.0005,total_stage)
rates=[]
for i in range(total_stage):
    scheduler.feed(lines)
    for j in range(lines):
        rates.append(scheduler.get_lr())
        scheduler.step()

# plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
print(rates)
plt.plot(np.arange(0,total_stage*lines),rates)
plt.pause(0)
c=0
