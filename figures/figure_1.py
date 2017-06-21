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

# Colors
c1  = (1, 1./3, 1./3, 1)
c2  = (1./3, 1./3, 1, 1)
c3  = (1./3, 1, 1./3, 1)
c12 = (2./3, 1./3, 2./3, 1)
c23 = (1./3, 2./3, 2./3, 1)
c13 = (2./3, 2./3, 1./3, 1)

Fx = (1/sqrt(2))*array([[0,1,0],[1,0,1],[0,1,0]])
Fz = array([[1,0,0],[0,0,0],[0,0,-1]])

dqr = 0.1
ds = 101
qmagic = 0.348

qr = arange(0.01, 1.01, dqr)
# qr[3] = qmagic
qr = insert(qr, 4, qmagic)
nqr = len(qr)
delta = linspace(-2, 2, ds)
opacity = zeros([nqr])

# make opacity graident
for i, q in enumerate(qr):
	if q < qmagic:
		opacity[i] = q/qmagic
	else:
		opacity[i] = 1-(q - qmagic)/(1-qmagic)

Hlist = [[2*pi*(Fx+q*Fz**2 +d*Fz) for d in delta] for q in qr]
evals = [[sort(linalg.eigvals(Hlist[i][j])/(2*pi)) for j in range(ds)] for i in range(nqr)]

e1 = zeros([nqr, ds])
e2 = zeros([nqr, ds])
e3 = zeros([nqr, ds])

# reshape into energy curves
for i in range(nqr):
	for j in range(ds):
		e1[i,j] = evals[i][j][0]-qr[i]
		e2[i,j] = evals[i][j][1]-qr[i]
		e3[i,j] = evals[i][j][2]-qr[i]

w12 = zeros([nqr, ds])
w23 = zeros([nqr, ds])
w13 = zeros([nqr, ds])

# make splitting curves
for i in range(nqr):
		w12[i,:] = abs(e1[i,:] - e2[i,:])
		w23[i,:] = abs(e2[i,:] - e3[i,:])
		w13[i,:] = abs(e1[i,:] - e3[i,:])

# Make figure
plt.figure(figsize=(9, 4.787))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1]) 
linw = 0.75
ax0 = plt.subplot(gs[0])
for i, q in enumerate(qr):
	plt.hlines(-qr[i], -2, 2, color='0.75', linestyle='--', alpha=opacity[i], lw=linw)
	plt.plot(delta, e1[i,:], color=c1, alpha=opacity[i], lw=(linw if q != qmagic else 1.5*linw))
	plt.plot(delta, e2[i,:], color=c2, alpha=opacity[i], lw=(linw if q != qmagic else 1.5*linw))
	plt.plot(delta, e3[i,:], color=c3, alpha=opacity[i], lw=(linw if q != qmagic else 1.5*linw))

plt.plot(delta, delta, color='0.75', linestyle='--', alpha=0.5, lw=linw)
plt.plot(delta, -delta, color='0.75', linestyle='--', alpha=0.5, lw=linw)
# plt.vlines(0, -2, 2, 'k', lw=0.5)
# plt.hlines(0, -2, 2, 'k', lw=0.5)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.ylabel(r"($\omega_{n}-q)/\Omega$")
plt.xlabel(r"$\Delta/\Omega$")
# Make labels
ax0.text(0.4, -1.1, r'$\mid 1 \rangle$', color=c1)
ax0.text(1.0,  0.1, r'$\mid 2 \rangle$', color=c2)
ax0.text(0.4,  1.3, r'$\mid 3 \rangle$', color=c3)
ax0.tick_params(direction='in')

# Make energy splitting figure
ax1 = plt.subplot(gs[1])
for i, q in enumerate(qr):
	plt.plot(delta, w23[i,:], color=c23, alpha=opacity[i], lw=(linw if q != qmagic else 1.5*linw))
	plt.plot(delta, w12[i,:], color=c12, alpha=opacity[i], lw=(linw if q != qmagic else 1.5*linw))
	plt.plot(delta, w13[i,:], color=c13, alpha=opacity[i], lw=(linw if q != qmagic else 1.5*linw))

plt.xlim(0,2)
plt.ylim(0,4)
plt.ylabel(r"$\omega_{ij}/\Omega$")
plt.xlabel(r"$\Delta/\Omega$")
plt.tight_layout(pad=0.2)
plt.subplots_adjust(wspace=.2)
ax1.text(0.1,1.7, r'$\omega_{12}$', color=c12)
ax1.text(0.1,0.4, r'$\omega_{23}$', color=c23)
ax1.text(0.1,2.6, r'$\omega_{13}$', color=c13)
ax1.tick_params(direction='in')

plt.savefig('figure_1.pdf')
plt.show()
