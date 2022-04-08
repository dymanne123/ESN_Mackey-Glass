import random
import numpy as np
from bayes_opt import BayesianOptimization
from reservoir_computing import rc
import matplotlib.pyplot as plt
import pandas as pd
random.seed(10)

r_c=rc(10000)
#X=r_c.sampling()
#r_c.training(X)
r_c.change_warm(100)
r_c.update_Win(1)
r_c.update_Wres(1.5,0.15)
X=r_c.sampling()
mse,mse1=r_c.training_online(X,0.00001,9000,num=1000)
print(mse,mse1)

