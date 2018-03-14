3.404 inches wide

plt.rcParams['lines.linewidth'] = 0.4 # 1.2
plt.rcParams['lines.markeredgewidth'] = 0.2 # 0.5
plt.rcParams['lines.markersize'] = 2.3 # 6.0
plt.rcParams['xtick.major.size'] = 1.5 # 4.0
plt.rcParams['ytick.major.size'] = 1.5 # 4.0
legend_keys = [x for x in plt.rcParams.keys() if 'legend' in x.lower() and type(plt.rcParams[x]) is float and 'scale' not in x and 'font' not in x]
# 'legend.borderaxespad' 0.5
# 'legend.borderpad'     0.4
# 'legend.columnspacing' 2.0
# 'legend.handleheight'  0.7
# 'legend.handlelength'  2.0
# 'legend.handletextpad' 0.8
# 'legend.labelspacing'  0.5
# [0.5, 0.4, 2.0, 0.7, 2.0, 0.8, 0.5]

for x in legend_keys:
    plt.rcParams[x] *= 0.4