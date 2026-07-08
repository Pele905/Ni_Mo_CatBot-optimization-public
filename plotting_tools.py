import matplotlib.pyplot as plt
import numpy as np


def extract_ECSA(ECSA_dict, 
                 plot = True):

    cmap = plt.get_cmap("coolwarm")
    blue = cmap(0.0)
    red = cmap(1.0)

    scan_rates = []
    cap_init = []
    cap_final = []

    for scan_rate in sorted(ECSA_dict.keys()):
        if scan_rate >= 41:
            continue
        scan_rates.append(scan_rate)
        cap_init.append(ECSA_dict[scan_rate]["Cap current init [mA]"])
        cap_final.append(ECSA_dict[scan_rate]["Cap current final [mA]"])

    scan_rates = np.array(scan_rates)
    cap_init = np.array(cap_init)
    cap_final = np.array(cap_final)

    # Linear fits
    slope_init, intercept_init = np.polyfit(scan_rates, cap_init, 1)
    slope_final, intercept_final = np.polyfit(scan_rates, cap_final, 1)

    xfit = np.linspace(scan_rates.min(), scan_rates.max(), 200)

    if plot == True:
        # Larger figure
        fig, ax = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

        # Before
        ax[0].scatter(scan_rates, cap_init, s=140, color=blue)
        ax[0].plot(xfit, slope_init*xfit + intercept_init, color=blue, lw=3, label=f"Capacitance = {10 ** 6 * slope_init:.0f}" + r" $\mu$F")

        ax[0].set_title("Initial", fontsize=24)
        ax[0].set_xlabel("Scan rate (mV s$^{-1}$)", fontsize=22)
        ax[0].set_ylabel("Capacitive current (mA)", fontsize=22)
        ax[0].legend(fontsize=18)

        # After
        ax[1].scatter(scan_rates, cap_final, s=140, color=red)
        ax[1].plot(xfit, slope_final*xfit + intercept_final, color=red, lw=3, label=f"Capacitance = {10 ** 6 * slope_final:.0f}" + r" $\mu$F")

        ax[1].set_title("Final", fontsize=24)
        ax[1].set_xlabel("Scan rate (mV s$^{-1}$)", fontsize=22)
        ax[1].legend(fontsize=18)

        for a in ax:
            a.tick_params(axis="both", labelsize=20)
            a.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return slope_init, slope_final