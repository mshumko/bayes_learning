import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec 
import os

### LOAD TRACE DATA ###
try:
    trace = pd.read_csv('./data/linear_regression_scratch_trace.csv')
except FileNotFoundError:
    print('FileNotFoundError: Linear regression trace not found. Run'
        ' linear_regression.py with variables run_scratch and'
        ' save_scratch_trace set to True')
    exit()

### TRUE PARAMETERS ###
# Make a straight line
y = [500, 10]
x = np.linspace(0, 20)
y_obs = np.array(y[0] + y[1]*x, dtype=int)
y_obs_noise = np.random.poisson(y_obs)

### GENERATE SUBPLOTS ###
fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(2, 2)
ax = plt.subplot(gs[:, 0], xlim=(0, 20), ylim=(450, 700))
bx = plt.subplot(gs[0, 1], xlim=(-25, 25))
cx = plt.subplot(gs[1, 1], xlim=(2, 5))

### PLOT FORMATTING.
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
plt.suptitle(f'Linear regression from scratch\n'
            f'f(x) = y[0] + x*y[1] | true values '
            f'y[0] = {y[0]}, y[1] = {y[1]}')

ax.plot(x, y[0] + y[1]*x, c='r', zorder=3)
#ax.scatter(x, y_obs, c='b', zorder=4)
animate_line, = ax.plot([], [], lw=3)

### ANIMATION PARAMETERS ###
start_frame = 0 # What frame to start at
plot_memory = 20  # How many last 
frames = 100
trace_quantiles = trace.quantile([0.01, .99])
intercept_bins = np.linspace(*trace_quantiles['y0'])
slope_bins = np.linspace(*trace_quantiles['y1'])

def init():
    animate_line.set_data([], [])
    return animate_line,

def animate(i):
    # How many traces to jump at a time.
    if i < 2/3*frames:
        skip_frames = 200
        index = i*skip_frames
        
        # delete all but plot_memory number of lines.
        if len(ax.lines) > plot_memory + 1:
            ax.lines.pop(2) # Remove the 1st element since the 0th is the true line.
    else:
        skip_frames = trace.shape[0]//(5*frames)
        index = trace.shape[0]-1

    # Plot another line
    ax.plot(x, trace.loc[index, 'y0'] + trace.loc[index, 'y1']*x, c='k', alpha=0.2)
    bx.clear(); cx.clear()
    bx.axvline(y[0], c='r')
    cx.axvline(y[1], c='r')
    bx.hist(trace.loc[:index, 'y0'], color='k', bins=intercept_bins)
    cx.hist(trace.loc[:index, 'y1'], color='k', bins=slope_bins)
    
    ### BX AND CX SUBPLOT FORMATTING ###
    bx.get_yaxis().set_ticks([]) #set_visible(False)
    cx.get_yaxis().set_ticks([]) #set_visible(False)
    bx.set_ylabel('posterior')
    bx.text(0.02, 0.98, r'$y[0]$', va='top', transform=bx.transAxes, fontsize=15)
    cx.set_ylabel('posterior')
    cx.text(0.02, 0.98, r'$y[1]$', va='top', transform=cx.transAxes, fontsize=15)
    return animate_line,

anim = FuncAnimation(fig, animate, init_func=init,
                    frames=frames, 
                    interval=100, blit=True)

if not os.path.exists('./plots/'): # Check if plots directory exists.
    os.makedirs('./plots/')
    print('Made a ./plots/ directory')
anim.save('./plots/linear_regression_from_scratch.gif', writer='imagemagick')