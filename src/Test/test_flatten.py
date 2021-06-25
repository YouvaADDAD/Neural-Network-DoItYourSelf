import numpy as np
from Flatten import Flatten

if __name__=="__main__":
    mod=Flatten()
    X=np.random.randn(4,4,4,4)
    print(mod.forward(X).shape)
    print(mod.backward_delta(X,mod.forward(X)).shape)