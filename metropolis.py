import numpy as np
import progressbar

import log_likelihood

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


### EXAMPLE ###
if __name__ == '__main__':
    print('Running Metroplis example.')
    import matplotlib.pyplot as plt
    import matplotlib
    import scipy.stats
    import functools

    # Make a straight line
    b = 2
    m = 10
    x = np.linspace(0, 10)
    y_obs = np.array(b + m*x, dtype=int)
    y_obs_noise = np.random.poisson(y_obs)

    # set up MCMC
    prior = [scipy.stats.uniform(0, 10), 
            scipy.stats.uniform(0, 100)]
    start = [prior_i.rvs() for prior_i in prior]

    def target(parameters, x, y_obs):
        """ Calculates the product of the likelihood and prior """
        y = parameters[0] + parameters[1]*x
        l = log_likelihood.log_poisson_likelihood(y, y_obs)
        p = np.sum([np.log(prior_i.pdf(p)) for prior_i, p in zip(prior, parameters)])
        return l + p

    def proposal(parameters, proposal_jump=[0.5, 0.5]):
        """ 
        Generate a new proposal, or "guess" for the MCMC to try next. 
        The new proposed value is picked from a Normal, centered on 
        the old value given by p. The proposal_jump array specifies 
        the standard deviation of the possible jumps from the prior
        parameter value.
        """
        new_vals = np.array([scipy.stats.norm(loc=p_i, scale=jump_i).rvs() 
                        for p_i, jump_i in zip(parameters, proposal_jump)])         
        return new_vals

    target_line = functools.partial(target, x=x, y_obs=y_obs)

    chain = log_metroplis(start, target_line, proposal, 5000, nburn=500)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].plot(chain[:, 0]); ax[1, 0].plot(chain[:, 1])
    ax[0, 1].hist(chain[:, 0]); ax[1, 1].hist(chain[:, 1])
    plt.show()