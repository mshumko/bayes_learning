print('Running linear regression from scratch example.')

import matplotlib.pyplot as plt
import scipy.stats
import functools
import numpy as np

from scratch import log_likelihood
from scratch import metropolis

# Make a straight line
y = [2, 10]
x = np.linspace(0, 10)
y_obs = np.array(y[0] + y[1]*x, dtype=int)
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

chain = metropolis.log_metroplis(start, target_line, proposal, 5000, nburn=500)

fig, ax = plt.subplots(nrows=2, ncols=2)
# Plot the chain and posteriors
ax[0, 0].plot(chain[:, 0]); ax[1, 0].plot(chain[:, 1])
ax[0, 1].hist(chain[:, 0], density=True)
ax[1, 1].hist(chain[:, 1], density=True)
# Overlay a vertical line over the true value 
ax[0,1].axvline(y[0], c='k')
ax[1,1].axvline(y[1], c='k')
# Lots of plot adjustments and labling
fig.suptitle(f'Linear regression from scratch\n'
            f'y=y[0] + x*y[1] | true values y[0] = {y[0]}, y[1] = {y[1]}')
ax[0,0].set(ylabel='y[0]', title='y[0] chain')
ax[1,0].set(ylabel='y[1]', xlabel='Iteration', title='y[1] chain')
ax[0,1].set(ylabel='probability', title='y[0] posterior')
ax[1,1].set(ylabel='probability', title='y[1] posterior')
# ax[0,1].axes.get_yaxis().set_visible(False)
# ax[1,1].axes.get_yaxis().set_visible(False)
plt.tight_layout(rect=(0,0,1,0.9))
plt.show()