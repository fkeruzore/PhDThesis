import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from chainconsumer import ChainConsumer
import os
from copy import copy
import fkplotlib


def rename_params(cc):
    for chain in cc.chains:
        for i, p in enumerate(chain.parameters):
            new_p = r"$" + p + r"$"
            new_p = new_p.replace("alpha", r"\alpha")
            new_p = new_p.replace("beta", r"\beta")
            new_p = new_p.replace("sigma", r"\sigma")
            new_p = new_p.replace(".YIZ", r"_{Y|Z}")
            new_p = new_p.replace(".XIZ", r"_{X|Z}")
            new_p = new_p.replace(".0", "")
            chain.parameters[i] = new_p


def get_data(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        os.system(
            f"scp keruzore@lpsc-nika2e.in2p3.fr:/data2e/keruzore/Fit_scaling/{path}/medians.csv"
            + f" /Users/keruzore/Fit_scaling/{path}/medians.csv"
        )
        os.system(
            f"scp keruzore@lpsc-nika2e.in2p3.fr:/data2e/keruzore/Fit_scaling/{path}/stdevs.csv"
            + f" /Users/keruzore/Fit_scaling/{path}/stdevs.csv"
        )


def summarize(paths, names, truth, errb=True, cmap=None, marginalize_only=True):
    all_figs, all_cc = {}, {}
    if cmap is None:
        cmap = "Spectral_r" if len(names) <= 5 else "rainbow_r"
    figsize = (4.5, (0.5 * len(names) + 1) if errb else (0.25 * len(names) + 2))

    # ===== Fetch data ===== #
    for f in paths:
        islist = isinstance(f, list)
        if islist:
            for f2 in f:
                get_data(f2)
        else:
            get_data(f)

    # ===== Medians plot (corner or marginalized) ===== #
    cc = ChainConsumer()
    for f, name in zip(paths, names):
        islist = isinstance(f, list)
        if islist:
            medians = pd.concat(
                [
                    pd.read_csv(f"{f2}/medians.csv")[
                        ["alpha.YIZ", "beta.YIZ", "sigma.YIZ.0"]
                    ]
                    for f2 in f
                ]
            )
        else:
            medians = pd.read_csv(f"{f}/medians.csv")[
                ["alpha.YIZ", "beta.YIZ", "sigma.YIZ.0"]
            ]
        cc.add_chain(medians, name=name)
    rename_params(cc)
    cc.configure(
        cmap=cmap,
        kde=False,
        smooth=0,
        linewidths=1.5,
        shade_gradient=0.0,
        shade_alpha=0.2,
        sigmas=[0, 1],
    )
    if marginalize_only:
        fig = cc.plotter.plot_summary(
            truth=[truth[p] for p in medians.columns], errorbar=errb
        )
        fig.set_size_inches(*figsize)
    else:
        fig = cc.plotter.plot(truth=[truth[p] for p in medians.columns])
        fig.set_size_inches(6, 6)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.align_labels()
    fig.suptitle("MCMC medians distributions")
    all_figs["medians"] = fig
    all_cc["medians"] = cc

    # ===== Bias & uncertainty quantification ===== #
    columns = []
    for p in ["alpha", "beta", "sigma"]:
        columns += [f"xi_{p}", f"zeta_{p}", f"eta_{p}"]

    all_quants = {c: [] for c in columns}
    cc_zeta, cc_xi, cc_eta = [ChainConsumer() for _ in range(3)]

    for f, name in zip(paths, names):
        islist = isinstance(f, list)
        if islist:
            meds = pd.concat([pd.read_csv(f"{f2}/medians.csv") for f2 in f])
            stds = pd.concat([pd.read_csv(f"{f2}/stdevs.csv") for f2 in f])
        else:
            meds = pd.read_csv(f"{f}/medians.csv")
            stds = pd.read_csv(f"{f}/stdevs.csv")
        zeta_chain, xi_chain, eta_chain = [
            {p: 0.0 for p in truth.keys()} for _ in range(3)
        ]

        for p1, p2 in zip(
            ["alpha.YIZ", "beta.YIZ", "sigma.YIZ.0"], ["alpha", "beta", "sigma"]
        ):
            # zeta = mean bias in sigmas
            zeta_chain[p1] = (meds[p1] - truth[p1]) / stds[p1]
            all_quants[f"zeta_{p2}"].append(np.mean(zeta_chain[p1]))

            # xi = mean bias in % of true value
            xi_chain[p1] = 100 * (meds[p1] - truth[p1]) / np.abs(truth[p1])
            all_quants[f"xi_{p2}"].append(np.mean(xi_chain[p1]))

            # eta = mean uncertainty in % of true value
            eta_chain[p1] = 100 * stds[p1] / np.abs(truth[p1])
            all_quants[f"eta_{p2}"].append(np.mean(eta_chain[p1]))
        cc_zeta.add_chain(zeta_chain, name=name)
        cc_xi.add_chain(xi_chain, name=name)
        cc_eta.add_chain(eta_chain, name=name)

    for cc, quant, label in zip(
        [cc_zeta, cc_xi, cc_eta], ["zeta", "xi", "eta"], ["Bias", "Bias", "Dispersion"]
    ):
        rename_params(cc)
        cc.configure(
            cmap=cmap,
            linewidths=1.5,
            statistics="mean",
            kde=False,
            smooth=0,
        )
        is_eta = quant == "eta"
        fig = cc.plotter.plot_summary(
            errorbar=errb, truth=None if is_eta else np.zeros(3)
        )
        fig.set_size_inches(*figsize)
        unit = "[\sigma]" if (quant == "zeta") else "[\%]"
        fig.suptitle(f"{label} $\\{quant} \; {unit}$")
        all_figs[quant] = fig
        all_cc[quant] = cc

    all_quants = pd.DataFrame(all_quants, index=names)

    return all_figs, all_cc, all_quants
