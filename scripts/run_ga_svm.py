import argparse
import os
import numpy as np
import pandas as pd

from src.features.ga_svm_selector import GASVMSelector
from src.metrics.time_memory import measure_time_and_memory
from src.metrics.energy_carbon import compute_energy, compute_carbon, compute_edp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--k", type=int, default=None)  
    parser.add_argument("--outdir", default="outputs/ga_svm/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X = pd.read_csv(args.X)
    y = pd.read_csv(args.y).values.ravel()

    selector = GASVMSelector(pop_size=20, num_generations=30)

    timing = measure_time_and_memory(selector.fit_transform, X, y)

    X_sel, mask, num_evals = timing["result"]

    wall = timing["WallTime(s)"]
    cpu = timing["CPUUtil_Avg(%)"]
    peak_mem = timing["PeakMem(MB)"]

    energy = compute_energy(cpu, wall)
    carbon = compute_carbon(energy)
    edp = compute_edp(energy, wall)

    np.save(f"{args.outdir}/ga_mask.npy", mask)

    df = pd.DataFrame([{
        "Method": "GA_SVM",
        "NumFeatures": mask.sum(),
        "NumEvals": num_evals,
        "WallTime(s)": wall,
        "CPUUtil(%)": cpu,
        "PeakMem(MB)": peak_mem,
        "Energy(J)": energy,
        "Carbon(gCO2e)": carbon,
        "EDP(J*s)": edp,
    }])
    df.to_csv(f"{args.outdir}/metrics_ga.csv", index=False)

    print(df)
    print("GA-SVM FS Completed.")


if __name__ == "__main__":
    main()
