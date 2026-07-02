import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(folder):
    """Laods the excels sheets out of the given folder, so it can be processed
    in other functions."""
    all_data = {}
    for file in folder.glob('*.xlsx'):
        sql_name = Path(file).stem
        all_data[sql_name] = pd.read_excel(file, sheet_name=0, index_col=0)
    return all_data

def load_std_data(folder):
    """
    Compute STD per variant by reading the individual per-variant sheets
    (which contain the 5 raw runs + an 'AVG' row) and excluding the AVG row.
    Returns {sql_name: DataFrame(index=variant, columns=metric)}.
    """
    all_std = {}
    for file in folder.glob('*.xlsx'):
        sql_name = Path(file).stem
        xl = pd.ExcelFile(file)
        variant_stds = {}
        for sheet in xl.sheet_names:
            if sheet == 'Averages':
                continue
            df = pd.read_excel(file, sheet_name=sheet, index_col=0)
            runs = df[df['Index'] != 'AVG']  # drop the AVG summary row
            variant_stds[sheet] = runs.std(numeric_only=True)
        all_std[sql_name] = pd.DataFrame(variant_stds).T
    return all_std

def print_avg_tables_latex(folder, output_folder):
    """Creates the tables in LaTeX format of the average measurements"""
    metrics = ['total times', 'n LLM calls', 'n input tokes', 'n output tokes']
    all_data = load_data(folder)
    all_std = load_std_data(folder)
 
    for metric in metrics:
        avg_rows = {}
        std_rows = {}
        for sql_name, df in all_data.items():
            if metric in df.columns:
                avg_rows[sql_name] = df[metric]
            std_df = all_std.get(sql_name)
            if std_df is not None and metric in std_df.columns:
                std_rows[sql_name] = std_df[metric]
 
        avg_pivot = pd.DataFrame(avg_rows).T
        std_pivot = pd.DataFrame(std_rows).T
 
        for drop_col in ['thalamusdb_combine', 'thalamusdb_LLM_descr_only_image']:
            avg_pivot = avg_pivot.drop(columns=drop_col, errors='ignore')
            std_pivot = std_pivot.drop(columns=drop_col, errors='ignore')
 
        # align columns/order and combine into "avg ± std" strings
        std_pivot = std_pivot.reindex(columns=avg_pivot.columns)
        combined = avg_pivot.copy().astype(object)
        for col in avg_pivot.columns:
            for idx in avg_pivot.index:
                mean_val = avg_pivot.loc[idx, col]
                std_val = std_pivot.loc[idx, col]
                if pd.isna(mean_val):
                    combined.loc[idx, col] = ''
                elif pd.isna(std_val):
                    combined.loc[idx, col] = f'{mean_val:.2f}'
                else:
                    combined.loc[idx, col] = f'{mean_val:.2f} $\\pm$ {std_val:.2f}'
 
        combined.index.name = 'SQL Query'
 
        # LaTeX code:
        latex_str = combined.to_latex(
            caption=f'{metric} (mean $\\pm$ std) per SQL query and optimizer variant',
            label=f'tab:{metric.replace(" ", "_")}',
            position='htbp',
        )
 
        filename = metric.replace(' ', '_') + '_table.tex'
        filepath = os.path.join(output_folder, filename)
        with open(filepath, 'w') as f:
            f.write(latex_str)
        print(f'Wrote {filepath}')
        print(latex_str)

def create_normalized_heatmaps(folder, output_folder):
    """Creates heatmaps of the average measurements in relation to the base case."""
    metrics = ['total times', 'n LLM calls', 'n input tokes', 'n output tokes']
    all_data = load_data(folder)

    # Build normalized pivots per metric
    pivots = {}
    for metric in metrics:
        rows = {}
        for sql_name, df in all_data.items():
            if metric in df.columns:
                rows[sql_name] = df[metric]

        pivot = pd.DataFrame(rows).T
        pivot.index.name = 'SQL Query'
        pivot = pivot.drop(columns='thalamusdb_combine', errors='ignore')
        pivot = pivot.drop(columns='thalamusdb_LLM_descr_only_image', errors='ignore')
        pivots[metric] = pivot.div(pivot['Base case'], axis=0)

    for metric in metrics:
        pivot_normalized = pivots[metric]

        _, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            pivot_normalized.drop(columns='Base case'),
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',
            center=1,
            linewidths=0.5,
            ax=ax
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        ax.set_title(f'{metric} — relative to Base case')
        plt.tight_layout()
        plt.savefig(output_folder / f'{metric.replace(" ", "_")}_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == '__main__':
    folder = Path(r'C:\Users\Mikev\study\scriptie\resluts\car_results\cars_p2')
    output_folder = Path(r'C:\Users\Mikev\study\scriptie\resluts\heatmaps_cars_p2')
    create_normalized_heatmaps(folder, output_folder)
    print_avg_tables_latex(folder, output_folder)