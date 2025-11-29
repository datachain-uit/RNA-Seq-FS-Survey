import argparse
import numpy as np
import pandas as pd
from classifiers.ml_classifiers import run_all_classifiers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs_method", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--fs_metrics", required=True)
    parser.add_argument("--X_train", required=True)
    parser.add_argument("--X_test", required=True)
    parser.add_argument("--y_train", required=True)
    parser.add_argument("--y_test", required=True)
    parser.add_argument("--outcsv", default="classification_results.csv")

    args = parser.parse_args()

    X_train = pd.read_csv(args.X_train)
    X_test = pd.read_csv(args.X_test)
    y_train = pd.read_csv(args.y_train).values.ravel()
    y_test = pd.read_csv(args.y_test).values.ravel()

    mask = np.load(args.mask)
    X_train_sel = X_train.loc[:, mask]
    X_test_sel = X_test.loc[:, mask]

    fs_info = pd.read_csv(args.fs_metrics).iloc[0].to_dict()
    fs_info["FS_Method"] = args.fs_method

    df = run_all_classifiers(
        X_train_sel, X_test_sel, y_train, y_test, fs_info
    )

    df.to_csv(args.outcsv, index=False)
    print("Saved:", args.outcsv)


if __name__ == "__main__":
    main()
