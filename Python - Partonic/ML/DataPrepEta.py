import pandas as pd
import numpy as np
import MC


def main():
    eta_total = 2000

    #Daten präparieren
    diff_WQ_eta = []
    eta = []
    diff_WQ = MC.diff_WQ_eta()
    eta_limit = 3
    save_path = "/home/andiw/Documents/Semester 6/Bachelor-Arbeit/pythonProject/Files/Partonic/PartonicData/"

    eta = np.linspace(start=-eta_limit, stop=eta_limit, num=eta_total)

    diff_WQ_eta = MC.gev_to_pb(diff_WQ(eta))

    diff_WQ_eta_data = pd.DataFrame(
        {
            "eta": eta,
            "WQ": diff_WQ_eta
        }
    )
    diff_WQ_eta_data.to_csv(save_path + "PlottingData2k", index=False)
    print(diff_WQ_eta_data)


if __name__ == "__main__":
    main()