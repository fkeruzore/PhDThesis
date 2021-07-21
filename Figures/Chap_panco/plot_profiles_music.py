import numpy as np
import matplotlib.pyplot as plt
import json
import os
import fkplotlib
import astropy.units as u
from astropy.cosmology import Planck18_arXiv_v2 as cosmo

fkplotlib.use_txfonts()
plt.ion()
path = "./all_profiles_music_covmat"

for f in os.listdir(path):
    plt.close("all")
    prof = np.load(f"{path}/{f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prof["r_music"], prof["p_music"], color="k", ls="--", label="MUSIC")
    ax.errorbar(
        prof["r_panco2"],
        prof["p_panco2"],
        yerr=prof["err_p_panco2"],
        color="tab:blue",
        fmt="o-",
        capsize=3,
        ms=3,
        label="PANCO2",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5 * prof["r_panco2"][0], 1.5 * prof["r_panco2"][-1])
    ax.set_ylim(
        0.8 * np.nanmin([prof["p_music"].min(), prof["p_panco2"].min()]),
        1.1 * np.nanmax([prof["p_music"].max(), prof["p_panco2"].max()]),
    )
    fkplotlib.ax_bothticks(ax)
    fkplotlib.ax_legend(ax, loc=1)
    ax.set_xlabel(r"Radius $r \;[{\rm kpc}]$")
    ax.set_ylabel(r"Pressure $P_{\rm e}(r) \;[{\rm keV \cdot cm^{-3}}]$")

    z = 0.54 if "0.54" in f else 0.82
    d_a = cosmo.angular_diameter_distance(z).to("kpc").value
    r_beam = prof["r_panco2"][0]
    r_500 = prof["r_panco2"][-2]
    r_fov = d_a * np.tan(6.5 * u.arcmin / 2)
    for r, label in zip([r_beam, r_500, r_fov], ["Beam", "$R_{500}$", "FoV"]):
        ax.axvline(r, 0, 1, ls=":", color="k", zorder=-5)
        ax.text(
            r,
            0.05,
            label,
            rotation=90,
            horizontalalignment="center",
            verticalalignment="bottom",
            bbox={"facecolor": "w", "edgecolor": "w"},
            transform=ax.get_xaxis_transform(),
            zorder=-4,
            fontsize=10.0,
        )
    fig.savefig(f"profiles_music_2/{f[:-4]}.pdf")
