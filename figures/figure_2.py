# make figure of long spectrogram

from numpy import sqrt, pi, argsort, 
import matplotlib.pyplot as plt
from matplotlib import gridspec
from faraday_aux import get_alazar_trace, load_stft, plot_stft
import pandas as pd
from lmfit.model import Model
from lmfit.models import SkewedGaussianModel, ConstantModel, GaussianModel
import peakutils
from peakutils.plot import plot as pplot


# Fit 7 skewed Gaussain functions using hack on multipeaks 
def fitNpeaks(f, y, npeaks=5, thres=0.02, min_dist=5, width=300,
			  plot_prefix=None, model=SkewedGaussianModel, offset=True):

	# Guess initial peak centres using peakutils
	indexes = peakutils.indexes(y, thres=thres, min_dist=min_dist)
	peaksfound = len(indexes)
	assert peaksfound >= npeaks, "Looking for %s or more peaks only found %s of them!" %(npeaks, peaksfound)
	peak_f = peakutils.interpolate(f, y, ind=indexes)
	
	# Oder peaks by decreaing height and keep only the first npeaks
	peak_heights = indexes
	peak_order = peak_heights.argsort()[::-1]
	peak_heights = peak_heights[peak_order[:npeaks]]
	peak_f = peak_f[peak_order[:npeaks]]
	amplitude_scale = 1.0
	peak_amplitudes = peak_heights * amplitude_scale       # This is lmfit's annoying definition of a Gaussian `amplitude'
	print('Initial peaks guessed at ', peak_f)

	# Make multipeak model
	peaks = []
	for i in range(npeaks):
		prefix = 'g{:d}_'.format(i+1)
		peaks.append(model(prefix=prefix))
		if i == 0:
			pars = peaks[i].make_params(x=f)
		else:
			pars.update(peaks[i].make_params())
		if model==SkewedGaussianModel:
			pars[prefix + 'center'].set(peak_f[i], min=f.min(), max=f.max())
			pars[prefix + 'sigma'].set(width, min=0.1*width, max=10*width)
			pars[prefix + 'gamma'].set(0, min=-5, max=5)
			pars[prefix + 'amplitude'].set(peak_amplitudes[i])
		elif model == GaussianModel:
			pars[prefix + 'center'].set(peak_f[i], min=f.min(), max=f.max())
			pars[prefix + 'sigma'].set(width, min=0.1*width, max=10*width)
			pars[prefix + 'amplitude'].set(peak_amplitudes[i])
	model = peaks[0]
	for i in range(1, npeaks):
		model += peaks[i]
	
	if offset:
		model+=ConstantModel()
		pars.update(ConstantModel().make_params())
		pars['c'].set(0)
	# Fit first spectrum, creating the ModelResult object which will be used over and over
	fit = model.fit(y, pars, x=f)
	return fit


# Set path of shot 

path = 'Z:\\Experiments\\spinor\\crossed_beam_bec\\2017\\05\\16\\20170516T172033_crossed_beam_bec_0.h5'

t, f, stft_whole, globs = load_stft(path, all_globals=True)
ti, V, globs = get_alazar_trace(path, tmax=1)

# Filter stft to ignore pre tip 
t_pulse = globs['hobbs_settle'] + globs['faraday_pre_tip_wait']+0.01
t_sub = t[t >= t_pulse][:-1]
stft_sub = stft_whole[t >= t_pulse][:-1]

# Sampling rate is 
fs = 20e6

# Create moving average of peaks
Pxx = (sum(stft_sub,0)/len(stft_sub[0]))
Pxx /=max(Pxx)

# Fit 7 peaks to moving average with gaussian model
fitG = fitNpeaks(f, Pxx, npeaks=7, thres=0.01, model=GaussianModel, offset=False)
paramsG = [fitG.best_values]
var_namesG = fitG.var_names
u_paramsG = [{var: fitG.params[var].stderr for var in var_namesG}]

Pxx_fitG = fitG.eval()

# Include offset for visual purposes
fitO = fitNpeaks(f, Pxx, npeaks=7, thres=0.01)
paramsO = [fitO.best_values]
var_namesO = fitO.var_names
u_paramsO = [{var: fitO.params[var].stderr for var in var_namesO}]

Pxx_fitO = fitO.eval()

# Fit N peaks using skewed Gaussian model
fit = fitNpeaks(f, Pxx, npeaks=7, thres=0.01, offset=False)
params = [fit.best_values]
var_names = fit.var_names
u_params = [{var: fit.params[var].stderr for var in var_names}]

Pxx_fit = fit.eval()

# Order peaks
center_keys = [key for key in fit.params.keys() if 'center' in key]
peakvals = sorted([fit.params[key].value for key in center_keys])
peakord = argsort([fit.params[key].value for key in center_keys])

# Get sigma values for skewed peaks and gauss peaks
sigmavalsG = [(fitG.params['g'+str(num+1)+'_sigma'].value, fitG.params['g'+str(num+1)+'_sigma'].stderr) for num in peakord]
sigmavals = [(fit.params['g'+str(num+1)+'_sigma'].value, fit.params['g'+str(num+1)+'_sigma'].stderr) for num in peakord]

# Calculate mean width of 23 peaks
avgsig23 = (sigmavalsG[1][0]+sigmavalsG[1][0])/2
u_avgsig23 = (sigmavalsG[1][1]+sigmavalsG[1][1])/2

print "Mean E23 peak width = %.1f(%.1f)" %(avgsig23, u_avgsig23)

# Calculate mean width of 12 peaks
avgsig12 = (sigmavals[2][0]+sigmavals[4][0])/2
u_avgsig12 = (sigmavals[2][1]+sigmavals[4][1])/2

print "Mean E12 peak width = %.1f(%.1f)" %(avgsig12, u_avgsig12)

# calculate mean skew of 12 peaks
skewvals = [(fit.params['g'+str(num+1)+'_gamma'].value, fit.params['g'+str(num+1)+'_gamma'].stderr) for num in peakord]
avgsk12 = (abs(skewvals[2][0])+abs(skewvals[4][0]))/2
u_avgsk12 = (skewvals[2][1]+skewvals[4][1])/2

print "Mean E12 skew = %.1f(%.1f)" %(avgsk12, u_avgsk12)

# make figure
plt.figure(figsize=(10,4))
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1]) 

ax0 = plt.subplot(gs[0])
plot_stft(stft_sub, t_sub, f, range=[3,5], cmap = 'gray_r')
# plt.ylim(f.min, 3.5325e3)
plt.ylabel('Frequency (kHz)')
plt.xlabel('time (s)')

ax1 = plt.subplot(gs[1],sharey=ax0)
plt.semilogx(Pxx, f/1e3, 'k.')
plt.semilogx(Pxx_fitO, f/1e3, 'r')
# plt.ylim(3.51e3, 3.5325e3)
plt.setp(ax1.get_yticklabels(), visible=False)
plt.xlim(8e-4,2)
plt.tight_layout()
plt.xlabel('Power (arb)')
plt.subplots_adjust(wspace=.0)

plt.show()