import pandas as pd
from collections import defaultdict


def load_annotations(summary_path):
    df = pd.read_csv(summary_path)
    sample_to_stage = dict(zip(df['Samples'], df['Stages']))
    return sample_to_stage


def select_samples_by_stage(matrix_path, sample_to_stage, target_per_stage=6250):
    selected_samples = []
    selected_stages = {}
    stage_counts = defaultdict(int)

    with open(matrix_path, 'r') as f:
        header = f.readline().strip().split('\t')
        gene_col, sample_cols = header[0], header[1:]

    for full_sample in sample_cols:
        parts = full_sample.split('_')
        if len(parts) < 3:
            continue

        short_sample = f"{parts[-2]}_{parts[-1]}"
        stage = sample_to_stage.get(short_sample)

        if stage and stage_counts[stage] < target_per_stage:
            selected_samples.append(full_sample)
            selected_stages[full_sample] = stage
            stage_counts[stage] += 1

        if all(count >= target_per_stage for count in stage_counts.values()):
            break

    return gene_col, selected_samples, selected_stages, stage_counts


def extract_selected_matrix(matrix_path, gene_col, selected_samples, selected_stages, output_path):
    usecols = [gene_col] + selected_samples

    df = pd.read_csv(matrix_path, sep="\t", usecols=usecols)
    df_T = df.set_index(gene_col).T
    df_T.index.name = "Sample"
    df_T["Stage"] = df_T.index.map(selected_stages)
    df_T.to_csv(output_path)
