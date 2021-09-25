import pandas as pd
import numpy as np
from scipy.stats import norm
import tabulate
import matplotlib.pyplot as plt
#import seaborn as sns

#Black Scholes Class

class BlackScholes:

    def __init__(self,spot,strike,r,sigma,tou):
        self.spot=spot
        self.strike=strike
        self.r=r
        self.tou=tou
        self.sigma=sigma
        self.d1=(np.log(self.spot/self.strike)+(self.r+0.5*self.sigma*self.sigma)*self.tou)/(self.sigma*np.sqrt(self.tou))
        self.d2=(np.log(self.spot/self.strike)+(self.r-0.5*self.sigma*self.sigma)*self.tou)/(self.sigma*np.sqrt(self.tou))
    
    def callPrice(self):
        call=self.spot*norm.cdf(self.d1)-self.strike*np.exp(-self.r*self.tou)*norm.cdf(self.d2)
        return call
    
    def putPrice(self):
        put=-self.spot*norm.cdf(-self.d1)+self.strike*np.exp(-self.r*self.tou)*norm.cdf(-self.d2)
        return put
    
    def binarycallPrice(self):
        bcall=np.exp(-self.r*self.tou)*norm.cdf(self.d2)
        return bcall
    
    def binaryputPrice(self):
        bput=np.exp(-self.r*self.tou)*(1-norm.cdf(self.d2))
        return bput
    