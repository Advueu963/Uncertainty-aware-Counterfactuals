import re

import pandas as pd

# Load CSV
df = pd.read_csv("time_adam_min_validity.csv", index_col=0)


# Helper function to format LaTeX entry
def format_latex_entry(value: str):
    match = re.match(
        r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)[ ]*\+/-[ ]*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        value.strip(),
    )
    if match:
        mean, std = match.groups()
        return f"${float(mean):.3g} \\pm {float(std):.3g}$"
    return value


# Apply formatting to all cells
latex_df = df.applymap(format_latex_entry)

# Generate LaTeX table
latex_table = latex_df.to_latex(
    escape=False, column_format="l" + "c" * len(latex_df.columns)
)

print(latex_table)
