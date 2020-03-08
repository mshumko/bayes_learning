import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats
import functools
import pandas as pd
import numpy as np
import pymc3
import os

from scratch import log_likelihood
from scratch import metropolis

run_scratch = True
save_scratch_trace = True
run_pymc3 = False

def polynomial(x, p):
    y = np.zeros_like(x)
    for i, p_i in enumerate(p):
        y += p_i * x**i
    return y

# Make a polynomial
y_params = [10, 8, 1]
x = np.linspace(-10, 10, 100)
y = polynomial(x, y_params)
errors = 10+x
y_obs = np.random.normal(loc=y, scale=errors)

# General MCMC parameters
niter = 30000
nburn = 1000

if run_scratch:
    ####################################
    ########### From scratch ###########
    ####################################
    print('Running linear regression from scratch example.')

    # set up MCMC
    prior = [scipy.stats.uniform(0, 30), 
            scipy.stats.uniform(0, 10), 
            scipy.stats.uniform(0, 3)]
    start = [prior_i.rvs() for prior_i in prior]

    def target(parameters, x, y_obs):
        """ Calculates the product of the likelihood and prior """
        y = polynomial(x, parameters)
        errors = 30+x # We assume larger errors.
        normal_likelihood_params = (y, errors)
        l = log_likelihood.log_normal_likelihood(y_obs, normal_likelihood_params)
        p = np.sum([np.log(prior_i.pdf(p)) for prior_i, p in zip(prior, parameters)])
        return l + p

    def proposal(parameters, proposal_jump=[1, 1, 1]):
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
    proposal_polynomial = functools.partial(proposal, proposal_jump=[1, 1, 0.1])

    # Run the MCMC sampler
    scratch_chain = metropolis.log_metroplis(start, target_line, proposal_polynomial, 
                                    niter, nburn=nburn)

    if save_scratch_trace:
        if not os.path.exists('./data/'): # Check if data directory exists.
            os.makedirs('./data/')
            print('Made a ./data/ directory')
        df = pd.DataFrame(scratch_chain, columns=['A', 'x0', 'sigma'])
        df.to_csv('./data/nonlinear_regression_scratch_trace.csv', index=False)

if run_pymc3:
    ####################################
    ########### Using pymc3 ############
    ####################################
    print('Running linear regression using pymc3 example.')

    with pymc3.Model() as model:
        # Define priors
        y0 = pymc3.Uniform('y0', 250, 600)
        y1 = pymc3.Uniform('y1', 0, 20)

        # Define likelihood
        likelihood = pymc3.Poisson('f(y)', mu=y0 + y1*x, observed=y_obs_noise)

        # Inference
        pymc_chain = pymc3.sample(draws=niter, cores=None, tune=nburn)

####################################
######## Plot the results ##########
####################################
if run_scratch:
    fig = plt.figure(constrained_layout=True, figsize=(8, 5))
    gs = gridspec.GridSpec(nrows=3, ncols=4, figure=fig)
    ax = np.array([
                    [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                    [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
                    [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
                ])
    bx = fig.add_subplot(gs[:, 2:])

    # Plot the chain and posteriors
    for i in range(scratch_chain.shape[1]):
        ax[i, 0].plot(scratch_chain[:, i]) 
        ax[i, 1].hist(scratch_chain[:, i], density=True)
        ax[i,1].axvline(y_params[i], c='k')

    # Superpose the 95% credible interval estimated from the posterior.
    ci = np.nan*np.ones((x.shape[0], 2))
    for i, x_i in enumerate(x):
        # y_scatter = gaus(scratch_chain[:, 0] + x_i*scratch_chain[:, 1]
        # y_scatter = scratch_chain[:, 0] + x_i*scratch_chain[:, 1]
        y_scatter = [polynomial(x_i, chain_i) for chain_i in scratch_chain]
        ci[i, :] = np.quantile(y_scatter, (0.025, 0.975))

    plt.plot(x, y, 'b', label='True')
    plt.scatter(x, y_obs, c='r', label='True+noise')
    bx.fill_between(x, ci[:, 0], ci[:, 1], color='g', alpha=0.3, label='95% CI')

    # Lots of plot adjustments and labling
    fig.suptitle(f'Linear regression from scratch\n'
                r'$f(x) = y_0 + y_1 x + y_2 x^2$ '
                f'| true values y[0] = {y_params[0]}, '
                f'y[1] = {y_params[1]}, y[2] = {y_params[2]}')
    ax[0,0].set(ylabel='y[0]', title='y[0] chain')
    ax[1,0].set(ylabel='y[1]', title='y[1] chain')
    ax[2,0].set(ylabel='y[2]', title='y[1] chain')
    ax[-1, 0].set(xlabel='Iteration')
    ax[0,1].set(ylabel='probability', title='y[0] posterior')
    ax[1,1].set(ylabel='probability', title='y[1] posterior')
    ax[2,1].set(ylabel='probability', title='y[2] posterior')
    bx.set(xlabel='x', ylabel='f(x)')
    bx.legend()

if run_pymc3: # Plot the pymc3 results for confirmation.
    # Use the bult in traceplot functionality to visualize the posteriors.
    lines = [('y0', {}, [y[0]]), ('y1', {}, [y[1]])] # This API is very cumbersome!
    pymc3.traceplot(pymc_chain[100:], lines=lines)

    # Now make a plot similar to the from scratch plot I made of the family
    # of lines picked from the posterior.
    plt.figure(figsize=(7, 7))
    plt.plot(x, y_obs_noise, 'x', label='True+noise')
    pymc3.plot_posterior_predictive_glm(pymc_chain, samples=100, eval=x,
                                lm=lambda x, sample: sample['y0'] + sample['y1']*x,
                                label='posterior predictive regression lines')
    plt.plot(x, y[0] + y[1]*x, label='True', lw=3., c='y')

    plt.title('Posterior predictive regression lines')
    plt.legend(loc=0)
    plt.xlabel('x')
    plt.ylabel('y')
    # Lastly, make a pairplot, i.e. a corner plot
    pymc3.plot_joint(pymc_chain, figsize=(5, 5), kind="hexbin")

plt.show()