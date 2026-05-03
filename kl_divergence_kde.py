import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ---------------------------------------------------
# 1. KL divergence between two samples (KDE-based)
# ---------------------------------------------------
def kl_divergence_continuous(p_samples, q_samples, grid_size=1000):
    p_samples = np.asarray(p_samples)
    q_samples = np.asarray(q_samples)

    kde_p = gaussian_kde(p_samples)
    kde_q = gaussian_kde(q_samples)

    xmin = min(p_samples.min(), q_samples.min())
    xmax = max(p_samples.max(), q_samples.max())

    x = np.linspace(xmin, xmax, grid_size)

    p_pdf = kde_p(x)
    q_pdf = kde_q(x)

    eps = 1e-12
    p_pdf = np.clip(p_pdf, eps, None)
    q_pdf = np.clip(q_pdf, eps, None)

    kl = np.sum(p_pdf * np.log(p_pdf / q_pdf)) * (xmax - xmin) / grid_size
    return kl


# ---------------------------------------------------
# 2. Rolling KL divergence
# ---------------------------------------------------
def rolling_kl_divergence(series, window=50, step=1):
    series = np.asarray(series)

    kl_values = []
    time_points = []

    for i in range(window, len(series) - window, step):
        reference = series[i - window:i]
        current = series[i:i + window]

        kl = kl_divergence_continuous(reference, current)

        kl_values.append(kl)
        time_points.append(i)

    return pd.DataFrame({
        "time_index": time_points,
        "kl_divergence": kl_values
    })


# ---------------------------------------------------
# 3. Full plotting function (paper-style figure)
# ---------------------------------------------------
def plot_tsa_kl(series, result, threshold=None):
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # -----------------------------
    # Top: original time series
    # -----------------------------
    ax[0].plot(series, linewidth=1)
    ax[0].set_title("Time Series with Regime Shift Detection")
    ax[0].set_ylabel("Value")

    # -----------------------------
    # Bottom: KL divergence
    # -----------------------------
    x = result["time_index"]
    y = result["kl_divergence"]

    ax[1].plot(x, y, linewidth=1.5, label="KL Divergence")

    if threshold is None:
        threshold = y.mean() + 2 * y.std()

    ax[1].axhline(threshold, linestyle="--", color="red", label="Threshold")

    spikes = result[y > threshold]
    ax[1].scatter(spikes["time_index"], spikes["kl_divergence"], color="red")

    ax[1].set_xlabel("Time Index")
    ax[1].set_ylabel("KL Divergence")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------
# 4. Example simulation (regime change)
# ---------------------------------------------------
if __name__ == "__main__":

    np.random.seed(42)

    # regime 1: stable system
    t1 = np.random.normal(0, 1, 500)

    # regime 2: shifted distribution (change point)
    t2 = np.random.normal(2, 1.5, 500)

    series = np.concatenate([t1, t2])

    # compute KL divergence over time
    result = rolling_kl_divergence(series, window=50)

    # plot full TSA figure
    plot_tsa_kl(series, result)
    
    