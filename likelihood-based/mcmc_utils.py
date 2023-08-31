import numpy as np
import pandas as pd


def inverse_variance_mean(frequency, data, error):
    """
    This function performs an inverse variance weighted mean of the frequency
    and of the data. The error is the standard error of the weighted mean

    Parameters
    ----------
    frequency, data, error : np.arrays (of floats)

    Returns
    -------
    mean_frequency, mean_data, mean_error : np.arrays (of floats)

    """

    if len(error) == 0:
        print("error is empty ...")

    else:
        mean_frequency = np.sum(frequency / error**2, axis=0) / np.sum(
            1 / error**2, axis=0
        )
        mean_data = np.sum(data / error**2, axis=0) / np.sum(1 / error**2, axis=0)
        mean_error = np.sum(1 / error**2, axis=0) ** (-1 / 2)

    return mean_frequency, mean_data, mean_error


def coarse_grain_vector(x, y, ngrid, std_y=None, w_y=None, mode="linear"):
    """
    Coarse-grains a `(x,y)` vector to `ngrid` points in linear or log scale
    (depending on `mode`).
    If standard deviations `std_y` given, inverse-weights the bins with them.
    If weights given, sums them per bin.

    Parameters
    ----------
    x, y, ngrid, std_y=None, w_y=None

    Returns
    -------
    x_coarse_grained, y_coarse_grained, std_y_coarse_grained, w_coarse_grained

    """
    if mode.lower() == "linear":
        x_cg = np.linspace(x[0], x[-1], ngrid + 1)
    elif mode.lower() == "log":
        x_cg = np.logspace(np.log10(x[0]), np.log10(x[-1]), ngrid + 1)
    else:
        raise ValueError("mode must be linear|log")
    if x_cg[1] - x_cg[0] < x[1] - x[0]:
        raise ValueError("Coarsing `ngrid` too dense for given `x`.")
    x_cg = x_cg[1:]

    x_cg_new, y_cg, std_y_cg, w_cg, position = [], [], [], [], [0]

    for i in range(0, len(x_cg)):
        try:
            position.append(np.where(x >= x_cg[i])[0][0])
        except IndexError:
            position.append(len(x))
        # Create partition of the x array
        # containing the indices of x between which each of x_cg lies
        data_vec = y[position[i] : position[i + 1]]
        freq_vec = x[position[i] : position[i + 1]]
        error_vec = std_y[position[i] : position[i + 1]] if len(std_y) > 0 else []
        weights_vec = w_y[position[i] : position[i + 1]] if len(w_y) > 0 else []
        # Inverse-average-weight the y's and std_y's corresponding to each x_cg
        my_result = inverse_variance_mean(freq_vec, data_vec, error_vec)
        # Appending the results in the lists to return
        x_cg_new.append(my_result[0])
        y_cg.append(my_result[1])
        std_y_cg.append(my_result[2])
        w_cg.append(np.sum(weights_vec))

    return np.array(x_cg_new), np.array(y_cg), np.array(std_y_cg), np.array(w_cg)


def coarse_grain_data(frequency, data_all, std_all, ngrid):
    ##
    cg_data = {}

    # Find the lower and highest decade in the frequency series data
    declims = [np.ceil(np.log10(min(frequency))), np.ceil(np.log10(max(frequency)))]
    declims = np.arange(declims[0], declims[-1] + 1, 1)

    my_data = {"XX": data_all["XX"]}

    for k in my_data.keys():
        i_l = 0
        nnf = []
        nnd = []
        nnstd = []
        nnw = []

        for i in range(len(declims)):
            i_r = np.where(frequency <= 10 ** declims[i])[0][-1]
            f = frequency[i_l:i_r]
            d = my_data[k][i_l:i_r]
            std = std_all[k][i_l:i_r]
            w = d**0

            if len(f) > ngrid:
                nf, nd, nstd, nw = coarse_grain_vector(
                    f, d, ngrid=ngrid, std_y=np.sqrt(std), w_y=w
                )
            else:
                nf, nd, nstd, nw = f, d, std, w

            nnf = np.append(nnf, nf)
            nnd = np.append(nnd, nd)
            nnstd = np.append(nnstd, nstd)
            nnw = np.append(nnw, nw)

            i_l = i_r

        cg_data["frequency_" + k] = nnf
        cg_data[k] = nnd
        cg_data["std_" + k] = nnstd
        cg_data["weight_" + k] = nnw

    return cg_data


def get_cg_data(f_arr, data, temp_noise, ngrid):
    cg_data = pd.DataFrame(
        coarse_grain_data(f_arr, {"XX": data}, {"XX": temp_noise}), ngrid
    )
    return cg_data


def get_R(samples):
    c_shape = samples.shape
    ### Chain means
    c_mean = np.mean(samples, axis=0)
    ### Global mean
    g_mean = np.mean(c_mean, axis=0)
    ### Intra chain variance
    within = np.std(samples, axis=0, ddof=1) ** 2
    ### Variance between chains_eq
    between = c_shape[1] / (c_shape[1] - 1) * np.mean((c_mean - g_mean) ** 2, axis=0)

    ### Averaged variance
    W = np.mean(within, axis=0)

    ### This is the R (for all paramters)
    val = ((c_shape[0] - 1) / c_shape[0] * W + between * (1 + 1 / c_shape[1])) / W

    return np.mean(np.sqrt(val))
