import numpy as np

from MaxPool1d import MaxPool1D

if __name__=='__main__':
    outPut=np.random.rand(10,6,2)
    maxP=MaxPool1D(k_size=2, stride=1)
    res=maxP.forward(outPut)
    print(outPut)
    print('###################')
    print(maxP.backward_delta(outPut,res))