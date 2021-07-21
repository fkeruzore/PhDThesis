import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import fkplotlib

# plt.style.use("dark_background")
fkplotlib.use_txfonts()
plt.ion()


def a10(r):
    """
    A10 params for ACTJ0215
    """
    P0, rp, a, b, c = [6.10498005e-2, 6.75922137e2, 1.05100000, 5.49050000, 0.308100000]
    x = r / rp
    return P0 * x ** (-c) * (1.0 + x ** a) ** ((c - b) / a)


def interp_powerlaw(x, y, x_new, axis=0):
    """
    Interpolate/extrapolate a profile with a power-law by performing
    linear inter/extrapolation in the log-log space.

    Args:
        x (array): input x-axis.
        y (array): input `f(x)`.
        x_new (float or array): `x` value(s) at which to perform interpolation.

    Returns:
        (float or array): `f(x_new)`

    """

    w_nonzero = np.where(x > 0.0)

    log_x = np.log10(x[w_nonzero])
    log_y = np.log10(y[w_nonzero])

    interp = interp1d(
        log_x,
        log_y,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
        axis=axis,
    )
    y_new = 10 ** interp(np.log10(x_new))

    return y_new


r_gnfw = np.logspace(np.log10(50), np.log10(1500), 100)
r_np = np.logspace(2, 3, 8)
r_np_2 = np.concatenate(([r_gnfw[0]], r_np, [r_gnfw[-1]]))
p_np = 0.5 * a10(r_np)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(r_gnfw, a10(r_gnfw), color="tab:blue", label="gNFW profile", zorder=1)
"""
ax.plot(r_np_2[0], 0.5 * a10(r_np_2[0]), color="tab:red", zorder=2, ls="--")
ax.plot(r_np_2[1], 0.5 * a10(r_np_2[1]), color="tab:red", zorder=2, ls="--")
ax.plot(
    r_np,
    0.5 * a10(r_np),
    color="tab:red",
    ls="-",
    marker="o",
    label="Binned profile",
    zorder=3,
)
"""
ax.plot(r_np, p_np, "o-", zorder=3, color="tab:red", label="Binned profile")
ax.plot(r_np_2, interp_powerlaw(r_np, p_np, r_np_2), ":", zorder=2, color="tab:red")
ax.set_xscale("log")
ax.set_yscale("log")


fkplotlib.ax_bothticks(ax)
ax.set_xlabel(r"Radius $r$")
ax.set_ylabel(r"Pressure $P_{\rm e}(r)$")
ax.legend(frameon=False)
# fig.savefig("gnfw_np_dark.pdf")
# fkplotlib.ax_legend(ax)
fig.savefig("gnfw_np.pdf")
