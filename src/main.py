import pandas as pd
import numpy as np

import tabulate
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.stats import norm,ttest_ind
import datetime
import tqdm
import argparse

np.random.seed(1)

from MonteCarlo import MonteCarlo

#Initial parameters
# S0=100 #Current price of the underlying
# E=100 #Strike price
# r=0.05 #Constant Risk-free rate
# sigma=0.2 #Volatility
# T=1 #Time to expiry in yrs
# dt=0.01 #Time step used in the MC simulation
# N=100 #No. of simulations

def run(option,S,X,r,sigma,T,dt,N):
    if option=='binaryCall':
        px=MonteCarlo(S,X,r,sigma,T,dt,N).binarycallPrice()
    elif option=='binaryPut':
        px=MonteCarlo(S,X,r,sigma,T,dt,N).binaryputPrice()
    elif option=='Call':
        px=MonteCarlo(S,X,r,sigma,T,dt,N).callPrice()
    elif option=='Put':
        px=MonteCarlo(S,X,r,sigma,T,dt,N).putPrice()
    else:
        px=np.nan
    
    print(f'Price of {option}', np.round(px,4))

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--option",type=str)
    parser.add_argument("--S",type=float)
    parser.add_argument("--X",type=float)
    parser.add_argument("--r",type=float)
    parser.add_argument("--sigma",type=float)
    parser.add_argument("--T",type=float)
    parser.add_argument("--dt",type=float)
    parser.add_argument("--N",type=int)
    
    args=parser.parse_args()
    
    run(
        option=args.option,
        S=args.S,
        X=args.X,
        r=args.r,
        sigma=args.sigma,
        T=args.T,
        dt=args.dt,
        N=args.N,
        )
    
