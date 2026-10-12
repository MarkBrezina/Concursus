"""
Created on Wed Nov  1 23:23:08 2023
@author: Mark Daniel Balle Brezina

"""

## This is an implementation of a simple integrator for a generic
# jump-diffusion process. This includes for a Euler─Mayurama and a Milstein
# integration scheme. To use the Milstein scheme the derivative of the diffusion
# function needs to be given. Created by Leonardo Rydin Gorjão and Pedro G. Lind



import pandas as pd
from IPython.display import Markdown, display
import numpy as np

class SSP:
    """
    The SSP class of stochastic processes
    """
    def __init__(self, runs = 1, timeframe = 30):
        """
        The SSP class takes in number of runs and timeframe to store it for later use with stochastic processes
        """
        self.runs = runs
        self.timeframe = timeframe

    #Random walk
    
    def RW(self, _print_ = False):
        """
        From the SSP class, the number of runs and the timeframe has been indicated.
        By adding True or _print_ = True in the input, the user will have a printout 
        of the mathematical description for the process. For this stochastic process
        no additional information is needed and the output will be a dataframe with
        number of runs times timeframe
        
        Abbreviations
        -------------
        The function is synonomous with RandomWalk, randomwalk, Random_Walk, R_W, r_w, rw and RW
        
        Parameters
        ----------
        _print_: (True/False): True prints the mathematical description of the formula used in the simulation
        
        Returns
        -------
        A numpy array of simulated paths for the random walk process
        
        Examples
        --------
        >>> random_walk(False)
        array([[ 0], [ 1], [ 2]],[[ 0], [ -1], [ 0]])
        
        >>> random_walk(True)
        This stochastic process is the discrete random walk, with the equation:  𝑥𝑡=𝑥𝑡−1+𝜖𝑡, where 𝜖𝑡 is the Rademacher distribution at time t:  𝑓(𝑘)={1/21/2𝑖𝑓𝑘=1𝑖𝑓𝑘=−1
        array([[ 0], [ 1], [ 2]],[[ 0], [ -1], [ 0]])
        
        """
        if _print_ == True:
            equation = r'$x_t = x_{t-1} + \epsilon_t$'
            eq = r'$ f(k)=\left\{\begin{matrix}1/2 & if \;\;\; k=1 \\1/2 & if \;\;\; k=-1 \\\end{matrix}\right.$'
            display(Markdown((f'This stochastic process is the discrete random walk, with the equation: {equation}, where $\epsilon_t$ is the Rademacher distribution at time t: {eq}')))

        all_walks = []
        for i in range(self.runs) :
            random_walk = [0]
            for t in range(self.timeframe) :
                step = random_walk[-1]
                probability = np.random.randint(0,2)
                if probability == 0:
                    step = step + 1
                elif probability == 1:
                    step = step - 1
                random_walk.append(step)
            all_walks.append(random_walk)
        dataframe_allwalks = np.array(all_walks)
        walks = pd.DataFrame(dataframe_allwalks)
        return(walks) 
    
    #Wiener process 
    
    def WP(self, _print_ = False):
        """
       From the SSP class, the number of runs and the timeframe has been indicated.
       By adding True or _print_ = True in the input, the user will have a printout 
       of the mathematical description for the process. For this stochastic process
       no additional information is needed and the output will be a dataframe with
       number of runs times timeframe
       
       Abbreviations
       -------------
       The function is synonomous with WienerProcess, wienerprocess, Wiener_Process, W_P, w_p, wp and WP.
       
       Parameters
       ----------
       _print_: (True/False): True prints the mathematical description of the formula used in the simulation
       
       Returns
       -------
       A numpy array of simulated paths for the simple wiener process
       
       Examples
       --------
       >>> wiener_process(False)
       array([[ 0], [ 1], [ 2]],[[ 0], [ -1], [ 0]])
       
       >>> wiener_process(True)
       This stochastic process is the discrete wiener process, with the equation: 𝑥𝑡=𝑥𝑡−1+√dt⋅𝑁𝑑𝑖𝑠𝑡, where 𝑁𝑑𝑖𝑠𝑡 is the standard normal distribution and √d𝑡 is the squareroot of the time difference between steps
       array([[ 0], [ 1], [ 2]],[[ 0], [ -1], [ 0]])
       
        """
        if _print_ == True:
            equation = r'$x_t = x_{t-1} + \sqrt{dt} \cdot N_{dist}$'
            display(Markdown((f'This stochastic process is the discrete wiener process, with the equation: {equation}, where' +r'${N_{dist}}$'+' is the standard normal distribution and $\sqrt dt$ is the squareroot of the time difference between steps')))
       
        all_walks = []
        for j in range(self.runs):
            walk = np.zeros((self.timeframe))
            walk[0] = 1
            for i in range(1, self.timeframe):
                yi = np.random.normal()
                walk[i] = walk[i-1]+(yi)*np.sqrt(1/self.timeframe)
            all_walks.append(walk)
        dataframe_allwalks = np.array(all_walks)
        walks = pd.DataFrame(dataframe_allwalks)

        return(walks)
    
    #Arithmetic brownian motion
    def ABM(self, mu = 1, sigma = 0.25, _print_ = False):
        """
        From the SSP class, the number of runs and the timeframe has been indicated.
        By adding True or _print_ = True in the input, the user will have a printout 
        of the mathematical description for the process. For this stochastic process
        no additional information is needed and the output will be a dataframe with
        number of runs times timeframe
        
        Abbreviations
        -------------
        The function is synonomous with WienerProcess, wienerprocess, Wiener_Process, W_P, w_p, wp and WP.
        
        Parameters
        ----------
        _print_: (True/False): True prints the mathematical description of the formula used in the simulation
        
        Returns
        -------
        A numpy array of simulated paths for the simple wiener process
        
        Examples
        --------
        >>> wiener_process(False)
        array([[ 0], [ 1], [ 2]],[[ 0], [ -1], [ 0]])
        
        >>> wiener_process(True)
        This stochastic process is the discrete wiener process, with the equation: 𝑥𝑡=𝑥𝑡−1+√dt⋅𝑁𝑑𝑖𝑠𝑡, where 𝑁𝑑𝑖𝑠𝑡 is the standard normal distribution and √d𝑡 is the squareroot of the time difference between steps
        array([[ 0], [ 1], [ 2]],[[ 0], [ -1], [ 0]])
        
        """
        if _print_ == True:
            equation = r'$x_t = x_{t-1} + \mu \cdot dt + \sigma \cdot \sqrt{dt} \cdot N_{dist}$'
            display(Markdown((f'This stochastic process is the discrete wiener process, with the equation: {equation}, where' +r'${N_{dist}}$'+' is the standard normal distribution and $\sqrt dt$ is the squareroot of the time difference between steps')))
        
        dt = 1/self.timeframe
        dW = np.sqrt(dt)*np.random.randn(self.runs, self.timeframe)
        dS = mu*dt + sigma*dW
        
        dS = np.insert(dS, 0, 1, axis=1)
        walks = np.cumsum(dS, axis=1)
        walks = pd.DataFrame(walks)
        return(walks)
    
    #Exponential brownian motion
    def EBM(self, mu = 1, sigma=0.25, _print_ = False):
        """
        From the SSP class, the number of runs and the timeframe has been indicated.
        By adding True or _print_ = True in the input, the user will have a printout 
        of the mathematical description for the process. For this stochastic process
        x0 is needed as additional information and the output will be a dataframe 
        with number of runs times timeframe
        :param _print_: must be True or False
        :param x0: Is set to be 1, unless changed by the user.
        :param sigma: is set to be 0.25, unless changed by the user.
        """
        
        if _print_ == True:
            equation = r'$x_t = x_{t-1} \cdot e^{\mu-0.5\cdot \sigma^{2}} \cdot dt + \sigma \cdot \sqrt{dt} \cdot N_{dist}$'
            display(Markdown((f'This stochastic process is the discrete financial wiener process with drift, with the equation: {equation} where' + r'${N_{dist}}$'+' is the standard normal distribution and $e^{r-0.5\cdot \sigma^{2}}$ is the drift w.r.t the risk-free rate and the standard deviation')))
            
        dt = 1/self.timeframe
        walks = pd.DataFrame(np.zeros((self.runs, self.timeframe)))
        walks.iloc[:,0] = 1
        for i in range(len(walks)):
            for t in range(1, self.timeframe):
                walks.iloc[i,t] = walks.iloc[i, t - 1] * np.exp((mu - 0.5*sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn(1, 1))

        return(walks) 
    
    # Brownian Bridge
    def BB(self, a, b, sigma, _print_ = False):
        """
        From the SSP class, the number of runs and the timeframe has been indicated.
        By adding True or _print_ = True in the input, the user will have a printout 
        of the mathematical description for the process. For this stochastic process
        
        dS = ((b-X)/(T-t))*dt + sigma*dW
        Model can support useful variance reduction techniques for pricing derivative contracts using Monte-Carlo simulation, 
        such as sampling. Also used in scenario generation.
        Requires numpy, pandas and plotly.express
        alpha = start
        beta = end
        sigma = spread
        Returns the paths, S, for the Brownian Bridge using the Euler-Maruyama method
        """
        
        if _print_ == True:
            equation = r'$x_{t+1} = x_{t} + \frac{(\beta-x_t)}{(T-t+1)} + \sigma \cdot dW(t) \sqrt{dt} \cdot N_{dist}$'
            display(Markdown((f'This stochastic process is the discrete financial wiener process with drift, with the equation: {equation} where' + r'${N_{dist}}$'+' is the standard normal distribution and $e^{r-0.5\cdot \sigma^{2}}$ is the drift w.r.t the risk-free rate and the standard deviation')))
        
        dt = 1/self.timeframe
        dW = np.random.randn(self.timeframe, self.runs)
        walks = np.concatenate((a*np.ones((1, self.runs)),np.zeros((self.timeframe-1, self.runs)), b*np.ones((1, self.runs))), axis=0)
    
        for i in range(0, self.timeframe-1):
            walks[i+1,:] = walks[i,:] + (b-walks[i,:])/(self.timeframe-i+1) +sigma * np.sqrt(dt) * dW[i,:]
    
        return(walks)
    
    # Variance Gamma
    def VG(self, mu = 1, sigma = 0.5, rate = 0.05):
        """
        From the SSP class, the number of runs and the timeframe has been indicated.
        By adding True or _print_ = True in the input, the user will have a printout 
        of the mathematical description for the process. For this stochastic process
        """
        if _print_ == True:
            equation = r'$x_{t+1} = x_{t} + \frac{(\beta-x_t)}{(T-t+1)} + \sigma \cdot dW(t) \sqrt{dt} \cdot N_{dist}$'
            display(Markdown((f'This stochastic process is the discrete financial wiener process with drift, with the equation: {equation} where' + r'${N_{dist}}$'+' is the standard normal distribution and $e^{r-0.5\cdot \sigma^{2}}$ is the drift w.r.t the risk-free rate and the standard deviation')))
        
        dt = 1/self.timeframe
        dG = np.random.gamma(dt*rate, rate, (self.timeframe, self.runs))
        dW = mu*dG+sigma*np.random.randn(self.timeframe, self.runs)*np.sqrt(dG)
        dW = np.insert(dW, 0, 1, axis=0)
        walks = np.cumsum(dW, axis=0)
        walks = pd.DataFrame(walks).T
        return(walks)
    
    # Square root diffusion
    def SRD(self, theta = 0.5, mu = 1 ,sigma = 0.25, typ = 'Euler'):
        """
        From the SSP class, the number of runs and the timeframe has been indicated.
        By adding True or _print_ = True in the input, the user will have a printout 
        of the mathematical description for the process. For this stochastic process
        """
        if _print_ == True:
            equation = r'$x_{t+1} = x_{t} + \frac{(\beta-x_t)}{(T-t+1)} + \sigma \cdot dW(t) \sqrt{dt} \cdot N_{dist}$'
            display(Markdown((f'This stochastic process is the discrete financial wiener process with drift, with the equation: {equation} where' + r'${N_{dist}}$'+' is the standard normal distribution and $e^{r-0.5\cdot \sigma^{2}}$ is the drift w.r.t the risk-free rate and the standard deviation')))
        
        mu = mu + 1
        walkh = np.zeros((self.timeframe, self.runs))
        walks = np.zeros((self.timeframe, self.runs))
        walkh[0] = 1
        walks[0] = 1
        dt = 1/self.timeframe

        if typ.lower() =="euler":
            for t in range(1, self.timeframe):
                walkh[t] = (walkh[t - 1] +
                         theta * (mu - np.maximum(walkh[t - 1], 0)) * dt +
                         sigma * np.sqrt(np.maximum(walkh[t - 1], 0)) *
                         np.sqrt(dt) * np.random.standard_normal(self.runs))
            walks = np.maximum(walkh, 0)

        elif typ.lower() =="exact":
            for t in range(1, self.timeframe):
                df = 4 * mu * theta / sigma ** 2
                c = (sigma ** 2 * (1 - np.exp(-theta * dt))) / (4 * theta)
                nc = np.exp(-theta * dt) / c * walks[t - 1] 
                walks[t] = c * np.random.noncentral_chisquare(df, nc, size=self.runs)

        walks = pd.DataFrame(walks).T
        return(walks)
    
