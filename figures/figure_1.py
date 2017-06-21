# Make figure 1 dressed paper
from __future__ import division
from numpy import *
import matplotlib.pyplot as plt
from matplotlib import gridspec

# figure prelims
plt.rcParams['axes.linewidth'] = 1.2 #set the value globally
plt.rcParams['font.size'] = 20.0
plt.rcParams['legend.fontsize'] = 16.0 # 'large'
plt.rcParams['mathtext.fontset'] = 'stix' # 'cm'
plt.rcParams['font.family'] = 'STIXGeneral' # [u'serif']

Fx = (1/sqrt(2))*array([[0,1,0],[1,0,1],[0,1,0]])
Fz = array([[1,0,0],[0,0,0],[0,0,-1]])

dqr = 12
ds = 101
qmagic = 0.348

qr = linspace(0.01, 1.01,dqr)
delta = linspace(-2,2,ds)
colour =zeros([dqr])

# make colour graident
for i, q in enumerate(qr):
	if q<qmagic:
		colour[i] = 0.9*(q/qmagic)**4+0.1
	else:
		colour[i] = 0.9*(1-(q - qmagic)/(1-qmagic))**4+0.1

Hlist = [[2*pi*(Fx+q*Fz**2 +d*Fz) for d in delta] for q in qr]
evals = [[sort(linalg.eigvals(Hlist[i][j])/(2*pi)) for j in range(ds)] for i in range(dqr)]

e1 = zeros([dqr,ds])
e2 = zeros([dqr,ds])
e3 = zeros([dqr,ds])

# reshape into energy curves
for i in range(dqr):
	for j in range(ds):
		e1[i,j] = evals[i][j][0]-qr[i]
		e2[i,j] = evals[i][j][1]-qr[i]
		e3[i,j] = evals[i][j][2]-qr[i]

w12 = zeros([dqr,ds])
w23 = zeros([dqr,ds])
w13 = zeros([dqr,ds])

# make splitting curves
for i in range(dqr):
		w12[i,:] = abs(e1[i,:] - e2[i,:])
		w23[i,:] = abs(e2[i,:] - e3[i,:])
		w13[i,:] = abs(e1[i,:] - e3[i,:])

# Make figure
plt.figure(figsize=(9,4.787))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 
linw = 1.1
ax0 = plt.subplot(gs[0])
for i in range(dqr):
	plt.hlines(-qr[i], -2, 2, color='0.75', linestyle='--',alpha=colour[i], lw=linw)
	plt.plot(delta, e1[i,:], 'r', alpha=colour[i],lw=linw)
	plt.plot(delta, e2[i,:], 'b', alpha=colour[i],lw=linw)
	plt.plot(delta, e3[i,:], 'g', alpha=colour[i],lw=linw)

plt.plot(delta, delta, color='0.75', linestyle='--',alpha=0.75, lw=linw)
plt.plot(delta, -delta, color='0.75', linestyle='--',alpha=0.75, lw=linw)
plt.vlines(0, -2, 2, 'k', lw=0.5)
plt.hlines(0, -2, 2, 'k', lw=0.5)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.ylabel(r"($\omega_{n}-q)/\Omega$")
plt.xlabel(r"$\Delta/\Omega$")
# Make labels
ax0.text(0.4,-1.1, r'$\vert 1 \rangle$', color='r')
ax0.text(1.0,0.1, r'$\vert 2 \rangle$', color='b')
ax0.text(0.4,1.3, r'$\vert 3 \rangle$', color='g')
ax0.tick_params(direction='in')

# Make energy splitting figure
ax1 = plt.subplot(gs[1])
for i in range(dqr):
	plt.plot(delta, w23[i,:], color=(0.0,0.5,0.5,1), alpha=colour[i],lw=linw)
	plt.plot(delta, w12[i,:], color=(0.5,0.0,0.5,1), alpha=colour[i],lw=linw)
	plt.plot(delta, w13[i,:], color=(0.5,0.5,0.0,1), alpha=colour[i],lw=linw)

plt.xlim(0,2)
plt.ylim(0,4)
plt.ylabel(r"$\omega_{ij}/\Omega$")
plt.xlabel(r"$\Delta/\Omega$")
plt.tight_layout(pad=0.2)
plt.subplots_adjust(wspace=.2)
ax1.text(0.1,1.7, r'$\omega_{12}$', color=(0.5,0.0,0.5,1))
ax1.text(0.1,0.4, r'$\omega_{23}$',color=(0.0,0.5,0.5,1))
ax1.text(0.1,2.6, r'$\omega_{13}$', color=(0.5,0.5,0.0,1))
ax1.tick_params(direction='in')

plt.savefig('figure_1.pdf')
plt.show()
