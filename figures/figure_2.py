# make figure of long spectrogram

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from faraday_aux import get_alazar_trace, load_stft, plot_stft
import pandas as pd
from lmfit.model import Model
from lmfit.models import SkewedGaussianModel, ConstantModel, GaussianModel
from lmfit import Parameters
import peakutils
from peakutils.plot import plot as pplot
from analyse_dressed_spectroscopy import analyse_frequencies
from scipy import signal, stats
from scipy.linalg import expm


plt.rcParams['axes.linewidth'] = 1.2 #set the value globally
plt.rcParams['font.size'] = 20.0
plt.rcParams['legend.fontsize'] = 16.0 # 'large'
plt.rcParams['mathtext.fontset'] = 'stix' # 'cm'
plt.rcParams['font.family'] = 'STIXGeneral' # [u'serif']


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

# -- Gaussian fitting to get peak estimates --- #
# Create moving average of peaks
Pxx = (sum(stft_sub,0)/len(stft_sub[0]))
Pxx /=max(Pxx)

# Fit 7 peaks to moving average with gaussian model
npeaks = 7
fitG = fitNpeaks(f, Pxx, npeaks=npeaks, thres=0.01, model=GaussianModel, offset=False)
paramsG = [fitG.best_values]
var_namesG = fitG.var_names
u_paramsG = [{var: fitG.params[var].stderr for var in var_namesG}]


# Create data frame for params
df_params = pd.DataFrame(paramsG, index=t_sub)
df_u_params = pd.DataFrame(u_paramsG, index=t_sub)
center_freqs = df_params.filter(like='center').values[0]

# Turn periodogram into data frame
# cut data to ingnore frequency burps at start and finish
t_min = t_pulse+0.01
t_max = 0.12
V[ti<t_min] = 0
V[ti>t_max] = 0
freq, Pd = signal.periodogram(V, fs)
df = pd.DataFrame(Pd, index=freq, columns=['Pd'])

fband = 500
# filter between peaks based of Gaussian fits
momdf = pd.DataFrame(columns=('f1','f2','f3'))

for i, f0 in enumerate(center_freqs):
	fmin, fmax = (f0-fband/2, f0+fband/2)
	subdf = df.loc[fmin:fmax]
	# subdf.plot()
	# plt.show()
	f_sub = subdf.Pd.index
	y = subdf.Pd.values/sum(subdf.Pd.values)
	n = len(y)
	f1 = sum(y*f_sub)
	f2 = sum(y*(f_sub-f1)**2)
	f3 = sum(y*(f_sub-f1)**3)

	g1 = (np.sqrt(n*(n-1))/(n-2))*f3/(f2**(3/2))
	z1 = g1/np.sqrt((6*n*(n-1))/((n-2)*(n+1)*(n+3)))
	print z1

	momdf.loc[i] = [f1, np.sqrt(f2), np.cbrt(f3)]



# print out.fit_report()

# --------------- Use center frequencies to for SNR decay ---------------------- # 
ytick = range(3512,3534,4)
ymin, ymax = 3.510e6,3.533e6
# make figure
plt.figure(figsize=(9,4.787))
gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1]) 

ax0 = plt.subplot(gs[0])
plot_stft(stft_sub, (t_sub-t_sub.min())*1e3, f, range=[3,5], cmap = 'gray_r')
# plt.ylim(f.min, 3.5325e3)
plt.ylabel('frequency (kHz)')
plt.xlabel('time (ms)')
plt.xlim(0,90)
ax0.tick_params(direction='in')
plt.yticks(ytick)

ax1 = plt.subplot(gs[1],sharey=ax0)
plt.semilogx(df.Pd.values/df.Pd[ymin:ymax].max(), df.index/1e3, 'k', linewidth=0.5)
plt.ylim(ymin/1e3, ymax/1e3)
plt.setp(ax1.get_yticklabels(), visible=False)
plt.xlim(1e-4,2)
plt.xlabel('PSD')
plt.tight_layout(pad=0.2)
plt.subplots_adjust(wspace=.02)


# plt.savefig('figure3.pdf')

plt.show()