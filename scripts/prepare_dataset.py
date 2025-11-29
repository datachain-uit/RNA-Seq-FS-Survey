import argparse
from src.data.sample_selection import (
    load_annotations,
    select_samples_by_stage,
    extract_selected_matrix
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out", default="annotated_data_12500_samples_per_class.csv")
    parser.add_argument("--target", type=int, default=6250)
    args = parser.parse_args()

    sample_to_stage = load_annotations(args.summary)

    gene_col, selected_samples, selected_stages, counts = \
        select_samples_by_stage(args.matrix, sample_to_stage, args.target)

    print("Stage counts:", counts)

    extract_selected_matrix(
        args.matrix, gene_col, selected_samples, selected_stages, args.out
    )


if __name__ == "__main__":
    main()
