import pandas as pd
import numpy as np
from scipy.stats import norm

#Monte Carlo class
class MonteCarlo:
        
    def __init__(self,spot,strike,r,sigma,tou,dt,N):
        self.spot=spot
        self.strike=strike
        self.r=r
        self.tou=tou
        self.sigma=sigma
        self.dt=dt
        self.N=N
        self.n=int(self.tou/self.dt)
    
    def simulate(self):
        np.random.seed(1)
        S={}
        for k in range(self.N):
            S[k+1,0]=self.spot
            for i in range(self.n):
                #Euler method to simulate the entire path
                S[k+1,i+1]=S[k+1,i]+(self.r*S[k+1,i]*self.dt)+(np.sqrt(self.dt)*S[k+1,i]*self.sigma*norm.ppf(np.random.rand()))

        Sf=pd.DataFrame.from_dict(S,orient='index').rename(columns={0:'S'})
        Sf=Sf.reset_index()
        Sf['sim']=Sf['index'].apply(lambda x:x[0])
        Sf['t']=Sf['index'].apply(lambda x:x[1])
        Sf=Sf.pivot(index='t',columns='sim')['S']
        
        return Sf

    def plot(self):
        Sf=self.simulate()
        
        ax=Sf.plot(figsize=(25,10),legend=False,colormap='viridis')
        ax.set_xlabel('No. of timesteps',color='brown')
        ax.set_ylabel('Price of Underlying',color='brown')
        ax.margins(x=0)

    def payoffs(self):
        Sf=self.simulate()
        final=Sf.T.iloc[:,-1].reset_index()

        #Payoffs
        final['payoff_c']=final[self.n].apply(lambda x:max(x-self.strike,0))
        final['payoff_p']=final[self.n].apply(lambda x:max(self.strike-x,0))

        final['payoff_bc']=final[self.n].apply(lambda x:1 if x-self.strike>0 else 0)
        final['payoff_bp']=final[self.n].apply(lambda x:1 if x-self.strike<0 else 0)
        
        return final
    
    def callPrice(self):
        
        final=self.payoffs()
        call=final.payoff_c.mean()*np.exp(-self.r*(self.tou))
        
        return call

    def putPrice(self):
        
        final=self.payoffs()
        put=final.payoff_p.mean()*np.exp(-self.r*(self.tou))
        
        return put

    def binarycallPrice(self):
        
        final=self.payoffs()
        bcall=final.payoff_bc.mean()*np.exp(-self.r*(self.tou))
        
        return bcall
    
    def binaryputPrice(self):
        
        final=self.payoffs()
        bput=final.payoff_bp.mean()*np.exp(-self.r*(self.tou))
        
        return bput