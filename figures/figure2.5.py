# make figure of long spectrogram

from numpy import sqrt, pi
import matplotlib.pyplot as plt
from matplotlib import gridspec
from faraday_aux import get_alazar_trace, load_stft, plot_stft
import pandas as pd
from lmfit.model import Model
from lmfit.models import SkewedGaussianModel, ConstantModel
from lmfit import Parameters
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
		pars[prefix + 'center'].set(peak_f[i], min=f.min(), max=f.max())
		pars[prefix + 'sigma'].set(width, min=0.1*width, max=10*width)
		pars[prefix + 'gamma'].set(0, min=-5, max=5)
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

# Fit 7 peaks to moving average with skewed gaussian model
fit = fitNpeaks(f, Pxx, npeaks=7, thres=0.01)
params = [fit.best_values]
var_names = fit.var_names
u_params = [{var: fit.params[var].stderr for var in var_names}]

Pxx_fit = fit.eval()

# df_params = pd.DataFrame(params, index=t_sub)
# df_u_params = pd.DataFrame(u_params, index=t_sub)



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
plt.semilogx(Pxx_fit, f/1e3, 'r')
# plt.ylim(3.51e3, 3.5325e3)
plt.setp(ax1.get_yticklabels(), visible=False)
plt.xlim(8e-4,2)
plt.tight_layout()
plt.xlabel('Power (arb)')
plt.subplots_adjust(wspace=.0)

plt.show()