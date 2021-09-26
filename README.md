
# Monte Carlo approach to price Binary options


Monte Carlo simulation is one of the numerical methods which can be used to price options and is especially useful when closed-form solutions can be difficult to find and/or models/contracts are complex or difficult to evaluate. The Monte Carlo approach can simulate the entire path of the options and can be extremely efficient in incorporating complex path dependencies. The different steps used in the Monte Carlo approach to price options are outlined below:

1. The first step involves simulating the path of the underlying under a risk neutral random walk. Starting with the initial price S0 of the underlying, the idea is to arrive at a final price S using a discretized form of the stochastic differential equation for the underlying, also known as the Euler Maruyama method2. It's of the form: $\delta$S=rS$\delta$t+$\sigma$S$\sqrt{\delta t}$$\phi$, where $\phi$ is obtained from a standardized Normal distribution

2. The simulations are repeated N number of times.

3. The payoffs of the option under each simulation are calculated and averaged. In case of binary options, the payoff is calculated using the Heaviside function3, which gives an option pay-off of 1 if the option is in the money and 0 otherwise. It's mathematically represented as:

\begin{equation*} \mathcal{H}\left( x\right) :=\left{ \begin{array}{c} 1, \ 0, \end{array} \begin{array}{c} x>0 \ x<0 \end{array} \right. \end{equation*}

4. The average payoff is then discounted using the risk-free rate to get the option price.


# Monte Carlo approach to price Binary options

## How to use the codes in this repository

```python
pip install -r requirements.txt

cd src

To price an in-the-money Call option with the price of the underlying (S) at 100, the strike price (X) at 100, the risk-free rate (r) at 5%, the realized volatility (sigma) at 20%, the time to expiry of the option (T) at 1 year, the no. of simulations (N) at 100 and the timestep (dt) at 0.01, run the following code from the directory of the main.py file (src) 

python main.py --option Call --S 120 --X 100 --r 0.05 --sigma 0.2 --T 1 --dt 0.01 --N 1000

