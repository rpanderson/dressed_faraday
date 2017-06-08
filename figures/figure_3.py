# Run this after running analysislib.spinor.faraday.analyse_dressed_spectroscopy in an interactive session

from faraday_aux import load_stft, plot_stft
plt.rcParams['axes.linewidth'] = 1.2 #set the value globally
plt.rcParams['font.size'] = 20.0
plt.rcParams['legend.fontsize'] = 18.0 # 'large'
plt.rcParams['mathtext.fontset'] = 'stix' # 'cm'
plt.rcParams['font.family'] = 'STIXGeneral' # [u'serif']

c1  = (1, 1./3, 1./3)
c2  = (1./3, 1./3, 1)
c3  = (1./3, 1, 1./3)
c12 = (2./3, 1./3, 2./3)
c23 = (1./3, 2./3, 2./3)
c13 = (2./3, 2./3, 1./3)

def spectrogram_overlay(i, range=[2.75, 5.25], save_plot=True, show_plot=True, figsize=(9, 14),
                        dBmin=0e-3, dBmax=8e-3, tmin=0e-3, tmax=80e-3, xmin=30e-3, xmax=90e-3):
    df = all_df[i]
    u_df = all_u_df[i]
    frf = all_results[i]['frf']
    f0 = df.loc[xmin:xmin+1e-3].fL.values[0]+1*all_results[i]['q']
    results = all_results[i]

    # Composite figure
    plt.figure(figsize=figsize)
    
    # Calibration shot
    ax0 = plt.subplot(311)
    t, f, stft, globs = load_stft(os.path.join(expt_folder, shots[i][0]), all_globals=True)
    plot_stft(stft, t, f, cmap='gray_r', range=range)
    ((df.fL + df.q)/1e3).plot(ax=ax0, label=r'$f_{L}+q$', c='gold', yerr=u_df.fL/1e3)
    ((df.fL - df.q)/1e3).plot(ax=ax0, label=r'$f_{L}-q$', c='chocolate', yerr=u_df.fL/1e3)
    plt.axis(ymin=f0/1e3-6.25, ymax=f0/1e3+6.25)
    ax0.legend(shadow=True, numpoints=1, loc='lower right')
    plt.ylabel('frequency (kHz)')

    # Dressed plot
    ax1 = plt.subplot(312, sharex=ax0)
    t, f, stft, globs = load_stft(os.path.join(expt_folder, shots[i][1]), all_globals=True)
    plot_stft(stft, t, f, cmap='gray_r', range=range)
    # ax = plt.gca()
    ((df.f13_theory+frf)/1e3).plot(ax=ax1, label=r'$f_{\mathrm{rf}} + f_{13}$', c=c13, lw=2) #\,\mathrm{ (theory)}$')
    ((df.f12_theory+frf)/1e3).plot(ax=ax1, label=r'$f_{\mathrm{rf}} + f_{12}$', c=c12, lw=2) #\,\mathrm{(theory)}$')
    # ((df.f12+frf)/1e3).plot(ax=ax, yerr=all_u_df[i]/1e3, c='g', ecolor='g', label=r'$f_{12}\,\mathrm{(expt.)}$')
    ((df.f23_theory+frf)/1e3).plot(ax=ax1, label=r'$f_{\mathrm{rf}} + f_{23}$', c=c23, lw=2) #\,\mathrm{ (theory)}$')
    # ((df.f23+frf)/1e3).plot(ax=ax, yerr=all_u_df[i]/1e3, c='c', ecolor='c', label=r'$f_{23}\,\mathrm{(expt.)}$')
    plt.axhline(frf/1e3, ls='--', lw=2, c='orange', label=r'$f_{\mathrm{rf}}$')
    plt.xlabel('time (s)')
    plt.ylabel('frequency (kHz)')
    plt.axis(xmin=xmin, xmax=xmax, ymin=frf/1e3-0.1, ymax=frf/1e3+12.5)
    ds = shots[i][1].split('_')[0]
    # plt.title(r'{:}; $q_R$ = {:}'.format(ds, format_unc(all_results[i]['qR'], all_results[i]['u_qR'])))
    ax1.legend(shadow=True, numpoints=1, loc='upper right')

    # Parametric plot
    ax2 = plt.subplot(313)
    subdf = df.loc[tmin:tmax].dropna()
    u_subdf = u_df.loc[tmin:tmax].dropna()

    # Compute theoretical splittings -- with uniformly sampled dB (for plotting)
    dB_p = np.linspace(dBmin, dBmax, 200)
    B_p = all_results[i]['B0'] + dB_p
    fL_p = map(mean_splitting, B_p) 
    detuning_p = all_results[i]['frf'] - fL_p
    splittings_p = np.array([splittings(all_results[i]['q'], all_results[i]['fR'], delta) for delta in detuning_p])
    f12_p, f23_p, f13_p = splittings_p.T
    dB_mG = 1e3*subdf.dB.values
    u_dB_mG = 1e3*u_subdf.dB.values
    plt.errorbar(dB_mG, subdf.f23.values/1e3, xerr=u_dB_mG, yerr=u_subdf.f23.values/1e3, label=r'$f_{23}$ (expt.)', fmt='o', c=c23, ecolor=c23)
    plt.plot(1e3*dB_p, f23_p/1e3, label='$f_{23}$ (theory)', c=c23)
    plt.errorbar(dB_mG, subdf.f12.values/1e3, xerr=u_dB_mG, yerr=u_subdf.f12.values/1e3, label=r'$f_{12}$ (expt.)', fmt='o', c=c12, ecolor=c12)
    plt.plot(1e3*dB_p, f12_p/1e3, label='$f_{12}$ (theory)', c=c12)
    plt.xlabel(r'$\Delta B$ (mG)')
    plt.ylabel(r'splitting, $f_{ij}$ (kHz)')
    ax2.axis(xmin=1e3*dBmin, xmax=1e3*dBmax, ymin=3, ymax=8)
    # plt.title(plot_title)
    ax2.legend(shadow=True, numpoints=1, loc='lower right')
    plt.subplots_adjust(hspace=.1)
    plt.tight_layout(pad=0.2)

    if save_plot:
        plt.savefig('plots/spectrogram/{:}_spectrogram.png'.format(ds))
        plt.savefig('plots/spectrogram/{:}_spectrogram.pdf'.format(ds))
    if show_plot:
        plt.show()

    # Draw inset over 3.2mG range
    plt.figure(figsize=(3.71571,1.60502))
    ax = plt.subplot(111)
    plt.errorbar(dB_mG, subdf.f12.values/1e3, xerr=u_dB_mG, yerr=u_subdf.f12.values/1e3, label=r'$f_{12}$ (expt.)', fmt='o', c=c12, ecolor=c12)
    plt.plot(1e3*dB_p, f12_p/1e3, label='$f_{12}$ (theory)', c=c12)
    # ax.axis(xmin=0, xmax=3.2, ymin=5.475, ymax=5.545)
    ax.axis(xmin=0, xmax=3.2, ymin=5.485, ymax=5.535)
    ax.set_xticks([0, 1, 2, 3])
    # ax.set_yticks([5.48, 5.50, 5.52, 5.54])
    ax.set_yticks([5.49, 5.51, 5.53])
    ax.tick_params(left=False, top=False, labelbottom=False, labeltop=False, labelleft=False, labelright=True)
    plt.tight_layout(0.2)
    if save_plot:
        plt.savefig('plots/spectrogram/{:}_spectrogram_inset.png'.format(ds))
        plt.savefig('plots/spectrogram/{:}_spectrogram_inset.pdf'.format(ds))
    if show_plot:
        plt.show()