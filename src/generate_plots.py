import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import getdist.plots as plots
import getdist.mcsamples as mcsamples
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

rcParams['mathtext.default'] = 'regular'
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']


def main(output_path):
    os.makedirs(output_path, exist_ok=True)

    data = pd.read_parquet('./data/photometry_results.parquet')
    stars = data.query('ellipticity_i < 0.1').copy()
    galaxies = data.query('ellipticity_i >= 0.1').copy()

    # === Plot 1: Ellipticity histogram ===
    plt.figure(figsize=(8, 5))
    plt.hist(data['ellipticity_i'], bins=np.arange(0, 1, 0.05),
             color='green', alpha=0.3, label='Objects detected')
    plt.hist(data['ellipticity_i'], bins=np.arange(0, 1, 0.05),
             color='darkgreen', histtype='step', linewidth=2)
    plt.axvline(0.1, color='red', ls='--', label='Possible star/galaxy separation')
    plt.xlabel('Ellipticity (i-band)', fontsize=16)
    plt.ylabel(r'Counts $[0.05^{-1}]$', fontsize=16)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'ellipticity_hist.png'), dpi=400)
    plt.close()

    # === Plot 2: CModel vs PSF ===
    graph = sns.jointplot(
        x=stars['mag_i_gaussian'],
        y=stars['mag_i_cmodel'] - stars['mag_i_gaussian'],
        color='darkblue',
        label='Stars',
        kind='scatter'
    )
    sns.scatterplot(
        x=galaxies['mag_i_gaussian'],
        y=galaxies['mag_i_cmodel'] - galaxies['mag_i_gaussian'],
        color='darkred',
        label='Galaxies',
        ax=graph.ax_joint
    )
    sns.histplot(x=galaxies['mag_i_gaussian'], color='darkred',
                 ax=graph.ax_marg_x, fill=True)
    sns.histplot(y=galaxies['mag_i_cmodel'] - galaxies['mag_i_gaussian'],
                 color='darkred', ax=graph.ax_marg_y, fill=True)
    graph.ax_joint.set_xlabel('mag i-band (PSF)', fontsize=17)
    graph.ax_joint.set_ylabel(r'$\Delta$mag (CModel - PSF)', fontsize=17)
    graph.ax_joint.legend(fontsize=16, frameon=True,
                          facecolor='gray', framealpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'delta_mag_jointplot.png'), dpi=400)
    plt.close()

    # === Plot 3: RA/DEC Distribution ===
    fig, ax = plt.subplots(figsize=(10, 6))

    sc = ax.scatter(
        data['RA_i_psf'], 
        data['DEC_i_psf'], 
        s=20, 
        alpha=0.7,
        edgecolor='none',
        c=data['mag_i_gaussian'],
        cmap='viridis_r'
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Gaussian Magnitude (i-band)', size=14)

    ax.set_xlabel('RA [degrees]', fontsize=14)
    ax.set_ylabel('DEC [degrees]', fontsize=14)
    ax.set_title('Sky Distribution of Detected Objects', fontsize=16)

    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.7)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'ra_dec_distribution.png'), dpi=400)
    plt.close()

    # === Color indices ===
    bands = ['g', 'r', 'i', 'z', 'y']
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            b1, b2 = bands[i], bands[j]
            color_gaussian = f'{b1}{b2}_gaussian'
            color_cmodel = f'{b1}{b2}_cmodel'
            stars.loc[:, color_gaussian] = stars.loc[:, f'mag_{b1}_gaussian'] - stars.loc[:, f'mag_{b2}_gaussian']
            galaxies.loc[:, color_gaussian] = galaxies.loc[:, f'mag_{b1}_gaussian'] - galaxies.loc[:, f'mag_{b2}_gaussian']
            stars.loc[:, color_cmodel] = stars.loc[:, f'mag_{b1}_cmodel'] - stars.loc[:, f'mag_{b2}_cmodel']
            galaxies.loc[:, color_cmodel] = galaxies.loc[:, f'mag_{b1}_cmodel'] - galaxies.loc[:, f'mag_{b2}_cmodel']

    # === Plot 4: GetDist triangle plot ===
    labels = ['mag_i_gaussian', 'mag_i_cmodel',
              'gr_gaussian', 'ri_gaussian', 'iz_gaussian', 'zy_gaussian',
              'gr_cmodel', 'ri_cmodel', 'iz_cmodel', 'zy_cmodel']

    stars_filtered = stars[labels].replace([np.inf, -np.inf], np.nan).dropna()
    galaxies_filtered = galaxies[labels].replace([np.inf, -np.inf], np.nan).dropna()

    labels_tex = [lab.replace('_', r'\_') for lab in labels]

    samples_stars = mcsamples.MCSamples(
        samples=stars_filtered.values,
        names=labels,
        labels=labels_tex,
        label='Stars'
    )

    samples_galaxies = mcsamples.MCSamples(
        samples=galaxies_filtered.values,
        names=labels,
        labels=labels_tex,
        label='Galaxies'
    )

    samples_stars.updateSettings({'smooth_scale_2D': 0.5})
    samples_galaxies.updateSettings({'smooth_scale_2D': 0.5})

    g = plots.get_subplot_plotter()
    g.settings.lab_fontsize = 20
    g.settings.legend_fontsize = 28
    g.settings.alpha_filled_add = 0.6

    g.triangle_plot(
        [samples_stars, samples_galaxies],
        names=labels,
        filled=True,
        legend_labels=['Stars', 'Galaxies'],
        legend_loc='upper right',
        contour_colors=['C0', 'red'],
        contour_levels=[0.68, 0.95],
        show_titles=False
    )
    plt.savefig(os.path.join(output_path, 'triangle_plot.png'), dpi=500)
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_plots.py <output_path>")
        sys.exit(1)

    output_path = sys.argv[1]
    main(output_path)
