import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as patches

#import scripts.analysis_functions as analysis

def plot_overall_distribution(retrieval_dict, variable, ax, bins, dardar_filepath=None, factor=1):

    bins_centre = bins[:-1] + np.diff(bins) / 2

    if dardar_filepath != None:
        # load dardar distribution data
        dardar_iwp_dist_unnormed = pickle.load(
                open(dardar_filepath, "rb")
        )
        dardar_iwp_distribution, _ = np.histogram(
            bins[:-1]/factor, bins=bins/factor, weights=dardar_iwp_dist_unnormed/factor, density=True
        )
        ax.plot(
            bins_centre, dardar_iwp_distribution, color="gray", alpha=0.5, lw=2, ls="-"
        )
        ax.fill_between(
            bins_centre,
            dardar_iwp_distribution,
            0,
            color="lightgray",
            alpha=1,
            label="DARDAR",
        )

    database_iwp_distribution, _ = np.histogram(retrieval_dict[f"{variable}_true"]/factor, bins=bins/factor, density=True)
    pred_iwp_distribution, _ = np.histogram(
        retrieval_dict[f"{variable}_mean"]/factor, bins=bins/factor, density=True
    )
    #sample_iwp_distribution, _ = np.histogram(
    #    retrieval_dict["fwp_sample"], bins=fwp_bins, density=True
    #)

    ax.plot(bins_centre, database_iwp_distribution, color="C1", label="True")

    ax.plot(
        bins_centre, pred_iwp_distribution, color="C2", ls="-", label="Retrieval"
    )
    #ax.plot(
    #    bins_centre,
    #    sample_iwp_distribution,
    #    color="C0",
    #    ls="-",
    #    label="Retrieval sample",
    #)

    ax.grid(which="both", ls="dashed", color="grey", alpha=0.3)
    ax.legend(loc="upper right")

def plot_zonal_mean(
    retrieval_dict,
    variable,
    ax,
    bins,
):
    """
    Plot zonal mean of a given variable for database values, retrieval mean, retrieval sample, and dardar (pre-calculated).
    """

    bins_centre = bins[:-1] + np.diff(bins) / 2

    mean_database, _, _ = stats.binned_statistic(
        retrieval_dict["latitude"], retrieval_dict[f'{variable}_true'], statistic=lambda x: np.nanmean(x), bins=bins
    )
    mean_pred, _, _ = stats.binned_statistic(
            retrieval_dict["latitude"],
            retrieval_dict[f"{variable}_mean"],
            statistic=lambda x: np.nanmean(x),
            bins=bins,
    )
        #mean_sample, _, _ = stats.binned_statistic(
        #    retrieval_dict["latitude"],
        #    retrieval_dict[f"{variable}_sample"],
        #    statistic=lambda x: np.nanmean(x),
        #    bins=bins,
        #)

    ax.plot(mean_database, bins_centre, color="C1", label="Test set")
    ax.plot(mean_pred, bins_centre, color="C2", label="Retrieval")

    #ax.plot(mean_sample, bins_centre, color="C0", label="Retrieval sample")

    dardar_iwp_zonal_mean = pickle.load(
        open(f"../../DataStorage/DARDAR/DARDAR_IWP_zonalmean.pkl", "rb")
     )
    # normalise distributions
    for i in range(len(dardar_iwp_zonal_mean[:, 0])):
        if dardar_iwp_zonal_mean[i, 1] != 0:
            dardar_iwp_zonal_mean[i, 0] /= dardar_iwp_zonal_mean[i, 1]
            if dardar_iwp_zonal_mean[i, 0] > 100:
                dardar_iwp_zonal_mean[i, 0] = np.nan

    latitude_grid_dardar = np.linspace(-70, 70, 140)
    bins_centre_dardar = (
        latitude_grid_dardar[:-1] + np.diff(latitude_grid_dardar) / 2
    )
    ax.plot(
        dardar_iwp_zonal_mean[:-1, 0],
        bins_centre_dardar,
        color="gray",
        alpha=0.5,
        ls="-",
        label="DARDAR",
    )
    ax.fill_betweenx(
        bins_centre_dardar,
        dardar_iwp_zonal_mean[:-1, 0],
        0,
        color="lightgray",
        alpha=1,
    )

    ax.minorticks_on()
    ax.grid(which="both", ls="dashed", color="grey", alpha=0.3)
    if variable == "fwp":
        ax.set_xlabel("IWP [$\mathrm{kg \; m}^{-2}$]")
        ax.set_xlim([0.04, 0.29])
    ax.set_ylabel("Latitude [degrees]")


def plot_zonal_mean_weighted(
    retrieval_dict,
    variable,
    ax,
    bins,
    dardar_filepath):

    bins_centre = bins[:-1] + np.diff(bins) / 2

    dardar_zm_zonal_mean = pickle.load(
        open(dardar_filepath, "rb")
    )

    for i in range(len(dardar_zm_zonal_mean[:, 0])):
        if dardar_zm_zonal_mean[i, 1] != 0:
            dardar_zm_zonal_mean[i, 0] /= dardar_zm_zonal_mean[i, 1]
            if dardar_zm_zonal_mean[i, 0] > 8000:
                dardar_zm_zonal_mean[i, 0] = np.nan
            if dardar_zm_zonal_mean[i, 0] < 0:
                dardar_zm_zonal_mean[i, 0] = np.nan
                

    latitude_grid_dardar = np.linspace(-70, 70, 140)
    bins_centre_dardar = latitude_grid_dardar[:-1] + np.diff(latitude_grid_dardar) / 2
    ax.plot(
        dardar_zm_zonal_mean[:-1, 0],
        bins_centre_dardar,
        color="gray",
        alpha=0.7,
        ls="-",
        label="DARDAR",
    )

    mean_database = np.zeros(len(bins)-1)
    mean_pred = np.zeros(len(bins)-1)
    sample_pred = np.zeros(len(bins)-1)
    for lat_bin in range(len(bins)-1):
        lat_indices_db = np.where((retrieval_dict['latitude'] >= bins[lat_bin]) & (retrieval_dict['latitude'] < bins[lat_bin+1]))[0]

        selected_var_db = retrieval_dict[f'{variable}_true'][lat_indices_db]
        selected_iwp_db = retrieval_dict['fwp_true'][lat_indices_db]
        
        selected_iwp_db = np.where(selected_iwp_db > 10**(-2), selected_iwp_db, np.nan)
        mean_database[lat_bin] = np.nansum(selected_var_db*selected_iwp_db) / np.nansum(selected_iwp_db)


        lat_indices_pred = np.where((retrieval_dict['latitude'] >= bins[lat_bin]) & (retrieval_dict['latitude'] < bins[lat_bin+1]))[0]

        selected_var_pred = retrieval_dict[f'{variable}_mean'][lat_indices_pred]
        selected_iwp_pred = retrieval_dict[f'fwp_mean'][lat_indices_pred]
        selected_iwp_true = retrieval_dict[f'fwp_true'][lat_indices_pred]

        selected_iwp_pred = np.where(selected_iwp_true > 10**(-2), selected_iwp_pred, np.nan)
        mean_pred[lat_bin] = np.nansum(selected_var_pred*selected_iwp_pred) / np.nansum(selected_iwp_pred)
    
    ax.plot(mean_database, bins_centre, color='C1', label='Database')
    ax.plot(mean_pred, bins_centre, color='C2', label='Retrieval mean')
    ax.minorticks_on()
    ax.grid(which='both', ls='dashed', color='grey', alpha=0.3)
    ax.set_ylabel('Latitude [degrees]')


def plot_true_vs_retrieval(
    y_true,
    y_pred_mean,
    y_pred_quantiles,
    variable,
    ax,
    bins,
    plot_all_quantiles=False,
    color="black",
    label=None,
):
    """
    Plots (and calculates) the mean, median, quantiles of the retrieved *mean*, across the provided bins.
    """

    bins_centre = bins[:-1] + np.diff(bins) / 2
    # Compute some stats
    mean, _, _ = stats.binned_statistic(
        y_true, y_pred_mean, statistic=lambda x: np.nanmean(x), bins=bins
    )
    median, _, _ = stats.binned_statistic(
        y_true, y_pred_mean, statistic=lambda x: np.nanquantile(x, 0.50), bins=bins
    )
    if plot_all_quantiles == True:
        # this plots a line for each of the quantiles, rather than just 16% and 84%
        cmap = plt.get_cmap("YlOrRd_r")
        num_colors = 150
        colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]
        for i in range(y_pred_quantiles.shape[1]):
            median_q, _, _ = stats.binned_statistic(
                y_true,
                y_pred_quantiles[:, i],
                statistic=lambda x: np.nanmedian(x),
                bins=bins,
            )
            ax.plot(bins_centre, median_q, color=colors[i], lw=3, alpha=0.5)
    else:
        c_16, _, _ = stats.binned_statistic(
            y_true, y_pred_mean, statistic=lambda x: np.nanquantile(x, 0.16), bins=bins
        )
        c_84, _, _ = stats.binned_statistic(
            y_true, y_pred_mean, statistic=lambda x: np.nanquantile(x, 0.84), bins=bins
        )
        ax.plot(bins_centre, c_16, color=color, lw=3, ls="--", alpha=1.0)
        ax.plot(bins_centre, c_84, color=color, lw=3, ls="--", alpha=1.0)

    ax.plot(bins, bins, lw=4, color="lightgrey")

    (im,) = ax.plot(bins_centre, median, color=color, ls="-", lw=3, label=label)
    ax.plot(bins_centre, mean, color=color, ls=":", lw=3)

    #if variable != "zm" and variable != "dm":
    #    ax.legend(loc="upper left")

    ax.minorticks_on()
    ax.grid(which="both", ls="dashed", color="grey", alpha=0.3)

    ax.set_xlim([bins[0], bins[-1]])
    ax.set_ylim([bins[0], bins[-1]])

    return im


def calibration(y_pred_quantiles, y_true, quantiles):
    """
    Finds the fraction of cases that fall inside prediction intervals.
    These prediction intervals are formed by pairs of quantiles.

    For example:
    The 50% interval is the interval between the 0.25 and 0.75 quantile.
    This interval should contain the true value 50% of the time.
    (100% of the true cases should fall between quantiles 0.0 and 1.0).

    Interpretation:
    Curve above the diagonal: interval contains too many true values, model is overly cautious.
    """
    n_quantiles = len(quantiles)
    n_intervals = n_quantiles // 2

    # Pair quantiles: left = qs[i], right = qs[-(i+1)]
    # Example: q0<->q98, q1<->q97, q2<->q96 ...
    left_idxs  = np.arange(0, n_intervals)
    right_idxs = np.arange(n_quantiles - 1, n_quantiles - 1 - n_intervals, -1)

    # Compute interval widths (nominal coverages)
    # e.g., (0.99-0.01)=0.98, (0.98-0.02)=0.96, ...
    intervals = quantiles[right_idxs] - quantiles[left_idxs]

    fractions = np.zeros(n_intervals)

    for i in range(n_intervals):
        lower = y_pred_quantiles[:, left_idxs[i]]
        upper = y_pred_quantiles[:, right_idxs[i]]

        # Count truths inside this interval
        inside = (y_true >= lower) & (y_true <= upper)
        fractions[i] = inside.sum() / len(y_true)

    # Reverse so intervals go from small → large (same as quantnn)
    return intervals[::-1], fractions[::-1]


def calibration_type_2(y_pred_quantiles, y_true, quantiles):
    """
    Finds the fraction of true cases lying below the nth quantile.

    For example:
    50% of the true values should fall below the 0.5 quantile.
    100% of cases should fall below the 1.0 quantile.

    Interpretation
    Curve above the diagonal: too many true values lie below the quantile,
    the quantile is too high, model is overly cautious.

    Essentially, the main difference is that the second method looks at a single quantile and the number of true cases below it.
    The first method looks at the interval formed between two quantiles and the true values within this interval.
    
    """
    n_quantiles = len(quantiles)
    quantile_idxs = np.arange(0, n_quantiles + 1, 1)
    fractions = np.zeros(n_quantiles)
    for i in range(len(quantiles)):
        upper_quantile = y_pred_quantiles[:, quantile_idxs[i]]
        fractions[i] = len(np.where(y_true <= upper_quantile)[0]) / len(y_true)

    return fractions


def binned_mean_and_spread(y_true, y_pred, bins):
    """Mean and 16th/84th percentiles of y_pred, in bins of y_true."""
    def binned(stat):
        return stats.binned_statistic(y_true, y_pred, statistic=stat, bins=bins)[0]
    return (binned(np.nanmean),
            binned(lambda x: np.nanquantile(x, 0.16)),
            binned(lambda x: np.nanquantile(x, 0.84)))


def conditional_probability(H, bins_x, bins_y):

    # H = joint PDF
    # axis 0 (rows) = x
    # axis 1 (columns) = y

    dx = np.diff(bins_x)
    dy = np.diff(bins_y)

    # Get joint probability by multiplying by bin area
    P_xy = H * dx[:, None] * dy[None, :]

    # marginal:
    # p(x)=∫p(x,y)dy   i.e. sum over columns
    P_x = P_xy.sum(axis=1)

    # Conditional probability: Joint probability divided by marginal
    # p(y|x)=p(y,x)/p(x)
    P_y_given_x = P_xy / P_x[:, None]

    # Sanity check: sum over y for each x
    # print(np.sum(P_y_given_x, axis=1))

    return P_y_given_x


def plot_conditional_probability_2vars(x, y, bins, ax, vmin=1e-5, vmax=3e-3):
    # calculates and plots P(y|x)
    
    # p(x,y)
    H, xedge, yedge = np.histogram2d(x, y, bins=bins, density=True)
    Xedge, Yedge = np.meshgrid(xedge, yedge)
    #  Marginal probability: P(x) is the sum/integral of the probabilities for each value of y.
    # p(x)=∫p(x,y)dy

    # Conditional probability: Joint probability divided by marginal
    # p(y|x)=p(y,x)/p(x)
    
    bins_y = bins[1]
    bins_x = bins[0]
    dy = bins_y[1]-bins_y[0]
    dx = bins_x[1]-bins_x[0]
    
    # integral over y (Zm), which is columns of H
    marginal = np.sum(H*dy, axis=1)
    
    # columns of H are Zm
    # want columns to be IWP (i.e. x values)
    
    y_given_x = np.zeros_like(H.T)
    H = H.T
    
    for i in range(H.shape[0]):
        # iterate over rows (Zm)
        # now want to normalise each row by p(x)
        y_given_x[i] = H[i]/marginal
        
    c1 = ax.pcolormesh(Xedge, Yedge, y_given_x, norm=LogNorm(vmin=vmin, vmax=vmax))
    
    # check that ∫p(y)dy = 1
    # (1 column is a distribution of y for a given x)
    #for i in range(H.shape[1]):
    #    assert np.isclose(np.sum(y_given_x[:,i]*dy), 1)
    
    return y_given_x




"""
def add_statistics_box(
    y_true,
    y_pred_mean,
    bins,
    box_xy, # (x, y)
    box_size,  # (box_width, box_height)
    ax,
    text_y,
    text_x = None,
    ):

    if text_x is None:
        text_x = box_xy[0] + box_size[0]/10

    statistics = analysis.calculate_retrieval_statistics(
        y_pred_mean,
        y_true,
        bins,
        threshold = 1e-7)
    
    box = patches.Rectangle(
    (box_xy[0], box_xy[1]),
    box_size[0],
    box_size[1],
    linewidth=1,
    zorder=10,
    edgecolor="lightgray",
    facecolor="white",
    alpha=0.8,
    )
    
    ax.add_patch(box)

    ax.text(
        text_x,
        text_y[0],
        f'bias: {statistics["bias"]}',
        fontsize=24,
        zorder=11,
    )
    
    ax.text(text_x, text_y[1], f'$r$: {statistics["corr_coef"]}', fontsize=24, zorder=11)
"""