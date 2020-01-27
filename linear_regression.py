print('Running linear regression from scratch example.')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats
import functools
import numpy as np

from scratch import log_likelihood
from scratch import metropolis

# Make a straight line
y = [500, 10]
x = np.linspace(0, 20)
y_obs = np.array(y[0] + y[1]*x, dtype=int)
y_obs_noise = np.random.poisson(y_obs)

# set up MCMC
niter = 30000
nburn = 1000
prior = [scipy.stats.uniform(250, 600), 
        scipy.stats.uniform(0, 20)]
start = [prior_i.rvs() for prior_i in prior]

def target(parameters, x, y_obs):
    """ Calculates the product of the likelihood and prior """
    y = parameters[0] + parameters[1]*x
    l = log_likelihood.log_poisson_likelihood(y, y_obs)
    p = np.sum([np.log(prior_i.pdf(p)) for prior_i, p in zip(prior, parameters)])
    return l + p

def proposal(parameters, proposal_jump=[10, 0.5]):
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

target_line = functools.partial(target, x=x, y_obs=y_obs_noise)

# Run the MCMC sampler
chain = metropolis.log_metroplis(start, target_line, proposal, 
                                niter, nburn=nburn)

####################################
####### Now plot the results #######
####################################
fig = plt.figure(constrained_layout=True, figsize=(8, 5))
gs = gridspec.GridSpec(nrows=2, ncols=4, figure=fig)
ax = np.array([
                [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
            ])
bx = fig.add_subplot(gs[:, 2:])

# Plot the chain and posteriors
ax[0, 0].plot(chain[:, 0]); ax[1, 0].plot(chain[:, 1])
ax[0, 1].hist(chain[:, 0], density=True)
ax[1, 1].hist(chain[:, 1], density=True)
# Overlay a vertical line over the true value 
ax[0,1].axvline(y[0], c='k')
ax[1,1].axvline(y[1], c='k')

# Now plot the true line, superposed by the 95% credible interval
# estimated from the posterior.
ci = np.nan*np.ones((x.shape[0], 2))
for i, x_i in enumerate(x):
    y_scatter = chain[:, 0] + x_i*chain[:, 1]
    ci[i, :] = np.quantile(y_scatter, (0.025, 0.975))

bx.plot(x, y[0] + y[1]*x, 'b', label='True')
bx.scatter(x, y_obs_noise, c='r', label='True+noise')
bx.fill_between(x, ci[:, 0], ci[:, 1], color='g', alpha=0.3, label='95% CI')

# Lots of plot adjustments and labling
fig.suptitle(f'Linear regression from scratch\n'
            f'f(x) = y[0] + x*y[1] | true values y[0] = {y[0]}, y[1] = {y[1]}')
ax[0,0].set(ylabel='y[0]', title='y[0] chain')
ax[1,0].set(ylabel='y[1]', xlabel='Iteration', title='y[1] chain')
ax[0,1].set(ylabel='probability', title='y[0] posterior')
ax[1,1].set(ylabel='probability', title='y[1] posterior')
bx.set(xlabel='x', ylabel='f(x)')
bx.legend()
plt.show()