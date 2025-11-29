import argparse
import os
import numpy as np
import pandas as pd

from src.features.mutual_information import MutualInformationSelector
from src.metrics.time_memory import measure_time_and_memory
from src.metrics.energy_carbon import compute_energy, compute_carbon, compute_edp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--outdir", default="outputs/mutual_info/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X = pd.read_csv(args.X)
    y = pd.read_csv(args.y).values.ravel()

    selector = MutualInformationSelector(k=args.k)

    timing = measure_time_and_memory(selector.fit_transform, X, y)

    X_sel, mask, num_evals = timing["result"]
    wall = timing["WallTime(s)"]
    cpu = timing["CPUUtil_Avg(%)"]
    peak_mem = timing["PeakMem(MB)"]

    energy = compute_energy(cpu, wall)
    carbon = compute_carbon(energy)
    edp = compute_edp(energy, wall)

    np.save(f"{args.outdir}/mi_mask.npy", mask)

    df = pd.DataFrame([{
        "Method": "MutualInformation",
        "NumFeatures": mask.sum(),
        "NumEvals": num_evals,
        "WallTime(s)": wall,
        "CPUUtil(%)": cpu,
        "PeakMem(MB)": peak_mem,
        "Energy(J)": energy,
        "Carbon(gCO2e)": carbon,
        "EDP(J*s)": edp,
    }])
    df.to_csv(f"{args.outdir}/mi_metrics.csv", index=False)

    print("MI FS completed.")
    print(df)


if __name__ == "__main__":
    main()
