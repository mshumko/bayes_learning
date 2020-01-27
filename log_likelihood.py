import numpy as np
import math
import copy

def log_poisson_likelihood(y, y_obs):
    """
    Calculate the naural logarithm of the Poisson likelihood for
    modeled number of detections y and the observed number of 
    detections y_obs. The Poisson PMF is

    pmf(y_obs | y) = (y^(y_obs) * e^(-y))/ y_obs!

    Parameters
    ----------
    y : numpy array
        The population counts, also called lambda.
    y_obs : numpy array
        The observed counts, also called k.

    Returns
    -------
    log_likelihood : float
        The sum of of the Poisson log-likelihoods. 
    """
    assert len(y) == len(y_obs), ('The modeled and observed count'
                                ' lengths must be identical.')
    
    # Drop 0 y values because they diverge. The probability of 
    # any y_obs other than 0, given y is 0 is 0.
    y_ind = np.where(np.array(y) != 0)[0]
    y = y[y_ind]
    y_obs_copy = copy.copy(np.array(y_obs))
    y_obs_copy = y_obs_copy[y_ind]

    # Take the factorial and apply Stirling's approximation 
    # for y_obs > 100.
    log_factorial = [math.log(math.factorial(y_i)) if y_i < 100 
                    else y_i*np.log(y_i) - y_i for y_i in y_obs_copy]
    # Analytic derivation of the Poisson log-likelihood for a product 
    # of Poisson random variables
    log_likelihood = np.sum(y_obs_copy*np.log(y) - y - log_factorial)
    return log_likelihood

def log_norm_likelihood():
    """
    Calculate the naural logarithm of the Gaussian (norm) likelihood 
    for the modeled number of detections y and the observed number of 
    detections y_obs. The Poisson PMF is

    pmf(y_obs | y) = (y^(y_obs) * e^(-y))/ y_obs!

    Parameters
    ----------
    y : numpy array
        The population counts, also called lambda.
    y_obs : numpy array
        The observed counts, also called k.

    Returns
    -------
    log_likelihood : float
        The sum of of the Poisson log-likelihoods. 
    """
    return