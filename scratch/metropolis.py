import numpy as np
import progressbar

def log_metroplis(start, log_target, proposal, niter, nburn=0, 
            thin=1, verbose=False):
    """
    Implements the log Metropolis–Hastings sampler. The user must 
    specify the start model parameters, the log_target function 
    (the sum of the log likelihood and log prior 
    distributions), the proposal function to generate a new set 
    of model parameters will be generated, and niter number of 
    iterations. 

    Parameters
    ----------
    start : float array
        A set of starting model parameters. 
    log_target : callable
        A function that is the sum of the log likelihood and 
        log prior distributions.
    proposal : callable
        A function to generate a new set of model parameters 
        given the current parameters.    
    niter : int
        The number of Metropolis iterations. Watch out - if 
        nburn != 0 or thin != 1 then the length of the chain will 
        not be niter. It will be (niter-nburn)//thin.
    nburn : int, optional
        The number of burn-in steps.
    thin : int, optional
        Decimate the chain by only saving the thin(th)
        chain value. Useful to reduce the autocorrelation.
    verbose : bool, optional
        Print the current propsal values. Useful for debugging.

    Returns
    -------
    chain : array 
        The chain which is the posterior when histrogramed. 
        Length of (niter-nburn)//thin. 
    """
    niter = int(niter)
    current = start
    chain = np.nan*np.ones((niter, len(start)), dtype=float)

    for i in progressbar.progressbar(range(niter)):
        proposed = proposal(current)
        p = min(log_target(proposed)-log_target(current), 0)
        if np.log(np.random.random()) < p:
            current = proposed
        chain[i, :] = current
        if verbose:
            print(f'Current values {current}')
    return chain[int(nburn)::thin, :]