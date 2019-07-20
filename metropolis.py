import numpy as np
import progressbar

def metropolis(start, target, proposal, niter, nburn=0, 
            thin=1, verbose=False, log_likelihood=False):
    """
    This function implements the Metropolis–Hastings sampler.
    Does not use the log-Likelihood yet.
    """
    niter = int(niter)
    current = start
    post = -1E31*np.ones((niter, len(start)), dtype=float)

    # Determine if the target will be comparing the 
    # likelihood or log-likelihood.
    if log_likelihood:
        compare_value = 0
    else:
        compare_value = 1

    for i in progressbar.progressbar(range(niter)):
        proposed = proposal(current)
        p = min(target(proposed)/target(current), compare_value)
        if np.random.random() < p:
            current = proposed
        post[i, :] = current
        if verbose:
            print(f'Current values {current}')
    return post[int(nburn)::thin, :]
