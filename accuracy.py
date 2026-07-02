import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind_from_stats

def parse_score(val):
    """Changes values to a whole, so not percentages ('%') , but values between 0 and 1"""
    if type(val) == str:
        val = val.strip()
        if val.endswith('%'):
            return float(val.rstrip('%').strip()) / 100
        return 1.0  # 'No compare list' / 'nvt' -> trivial 100% match
    else:
        val = float(val)
        if val > 1:
            return val / 100
        else:
            return val

def create_accuracy_table(folder: Path) -> pd.DataFrame:
    """
    The function creates a new excels sheet from the excel sheets that
    are in the input folder, which creates a table where the apporaches
    are set as rows and queries as column, where the items are the
    accuracy of the average 'correct ids' column.
    
    Input:
    folder: a Path to a folder with the needed excel files
    """
    dfs = []
    dfs_std = []

    for file in folder.glob("*.xlsx"):
        query_name = file.stem

        df = pd.read_excel(file, sheet_name="Averages")
        df = df[["Unnamed: 0", "correct ids"]]
        df = df.rename(columns={"correct ids": query_name})
        dfs.append(df)

        xl = pd.ExcelFile(file)
        for method in xl.sheet_names:
            if method == 'Averages':
                continue
            df = xl.parse(method)
            runs = df[df['Index'] != 'AVG']
            scores = runs['correct ids'].apply(parse_score)
            dfs_std.append({'query': query_name, 'method': method, 'std': scores.std()})
    
    dfs_std = pd.DataFrame(dfs_std)
    std_table = dfs_std.pivot(index='method', columns='query', values='std')

    mean_table = dfs[0]

    for df in dfs[1:]:
        mean_table = mean_table.merge(df, on="Unnamed: 0", how="outer")

    mean_table = mean_table.set_index("Unnamed: 0")
    mean_table = mean_table.drop("thalamusdb_combine")
    std_table = std_table.drop("thalamusdb_combine")
    return mean_table, std_table

def create_metric_tables(folder: Path, metric: str):
    """
    metric: 'precision', 'recall', or 'f1 score'
    Returns (mean_df, std_df): method x query, computed across the 5 runs per sheet.
    """
    mean_records, std_records = [], []
    for file in sorted(folder.glob("*.xlsx")):
        query_name = file.stem
        xl = pd.ExcelFile(file)
        for sheet in xl.sheet_names:
            if sheet == 'Averages':
                continue
            df = xl.parse(sheet)
            runs = df[df['Index'] != 'AVG']
            scores = runs[metric].apply(parse_score)
            mean_records.append({'method': sheet, 'query': query_name, 'mean': scores.mean()})
            std_records.append({'method': sheet, 'query': query_name, 'std': scores.std()})
    mean_df = pd.DataFrame(mean_records).pivot(index='method', columns='query', values='mean')
    std_df = pd.DataFrame(std_records).pivot(index='method', columns='query', values='std')
    return mean_df, std_df

def print_latex(df: pd.DataFrame) -> None:
    """Function that prints the dataframe in LaTeX so it can be copied
    to Overleaf"""
    #Put the df in percentage format
    df = df * 100

    latex_table = df.to_latex(
        float_format="%.3f",
        index=True
    )

    print(latex_table)

METHOD_ORDER = ['Base case', 'thalamusdb_Batch', 'thalamusdb_LLM_descr',
                'thalamusdb_all_improvements', 'thalamusdb_earlier_join',
                'thalamusdb_embed_certain_rows']

def to_latex_table(mean_df, std_df, caption, label, query_order=None):
    """Translates dfs to LaTex format"""
    queries = query_order or mean_df.columns.tolist()
    header = " & ".join(f"\\rotatebox{{90}}{{{m.replace('_', chr(92)+'_')}}}" for m in METHOD_ORDER)
    lines = [
        r"\begin{table}[htbp]",
        r"    \begin{tabular}{l" + "r"*len(METHOD_ORDER) + "}",
        r"    \toprule",
        f"     & {header} \\\\",
        r"    SQL Query &  &  &  &  &  &  \\",
        r"    \midrule",
    ]
    for q in queries:
        cells = []
        for m in METHOD_ORDER:
            mean, std = mean_df.loc[m, q], std_df.loc[m, q]
            cells.append("--" if pd.isna(mean) else f"{mean:.2f} $\\pm$ {std:.2f}")
        lines.append(f"    {q.replace('_', chr(92)+'_')} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"    \end{tabular}",
              f"    \\caption{{{caption}}}", f"    \\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)

def create_and_to_latex_accuracy(folder) -> None:
    """Creates a accuracy tabel by calling create_accuracy_table and
    puts it in LaTeX (Overleaf) format so it can be copied directly to
    Overleaf
    
    Input:
    folder: a Path to a folder with the needed excel files
    """
    overview = create_accuracy_table(folder)
    print_latex(overview)

if __name__ == '__main__':
    folder = Path(r"C:\Users\Mikev\study\scriptie\resluts\car_results\cars_p2")
    save_name_sign = r"C:\Users\Mikev\study\scriptie\resluts\analyzed_results\significance_overview_cars.xlsx"

    specs = [
        ('precision', 'Precision (\\%, mean $\\pm$ std across 5 runs) per SQL query and optimizer variant.', 'tab:precision', False),
        ('recall',    'Recall (\\%, mean $\\pm$ std across 5 runs) per SQL query and optimizer variant.', 'tab:recall', True),
        ('f1 score',  'F1 score (\\%, mean $\\pm$ std across 5 runs) per SQL query and optimizer variant.', 'tab:f1', True),
    ]
    for metric, caption, label, add_note in specs:
        mean_df, std_df = create_metric_tables(folder, metric)
        cap = caption if add_note else caption
        print(to_latex_table(mean_df, std_df, cap, label))
        print()