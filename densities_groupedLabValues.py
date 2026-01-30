# current best
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

"""
The script processes a data frame of lab values and results in a density plot
based on grouped means.
Pseudocode:

chunk_size = 10
vals = sort(data values)

group_means = []

for i in range(0, length(vals), chunk_size):
    chunk = vals[i : i + chunk_size]

    if length(chunk) == chunk_size:

        group_means.append(mean(chunk))

density(group_means)


Workflow:
- Filter lab values by LOINC code
- Keep only positive values (QA)
- Sort lab values
- Group values into consecutive, non-overlapping chunks of exactly 10 values
- Exclude any incomplete chunk
- Compute mean per group
- Plot weighted KDE of group means (weights = 10)
- Export group summaries and plots into a ZIP file
"""

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_data(filepath):
    """
    Load the synthetic dataset with date parsing for relevant columns
    """
    df = pd.read_csv(
        filepath,
        parse_dates=[
            "Patient.birthDate",
            "Condition.recordedDate",
            "Observation.effective.x.extension.QuelleKlinischesBezugsdatum",
        ],
    )
    return df


# ---------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------
def filter_lab_data(df, loinc_codes, value_col):
    """
    Filter dataframe for specific LOINC codes and keep only positive values (QA)
    """
    filtered = df[df["Observation.code"].isin(loinc_codes)].copy()
    filtered = filtered[filtered[value_col] > 0]
    return filtered


# ---------------------------------------------------------------------
# Grouping logic: fixed chunks of 10
# ---------------------------------------------------------------------
def group_by_fixed_chunks(values, chunk_size=10):
    """
    Sort values and split into consecutive chunks of fixed size.
    Only chunks with exactly `chunk_size` elements are kept.
    """
    values = np.sort(values)
    n = len(values)

    groups = []
    # 0, 56, 10
    # 
    for i in range(0, n, chunk_size):
        chunk = values[i : i + chunk_size]
        if len(chunk) == chunk_size:
            groups.append(chunk)

    return groups


# ---------------------------------------------------------------------
# Density generation
# ---------------------------------------------------------------------
def generate_grouped_mean_density(df, value_col, title, xlabel):
    """
    Group values into fixed chunks of 10 and compute mean per group.
    Generate weighted KDE of group means.
    """
    values = df[value_col].dropna().values

    if len(values) < 10:
        return None, None

    groups = group_by_fixed_chunks(values, chunk_size=10)

    if len(groups) == 0:
        return None, None

    # Compute group means and counts
    group_summary = []
    for idx, g in enumerate(groups):
        group_summary.append(
            {
                "group_id": idx + 1,
                "group_mean": np.mean(g),
                "n_obs": len(g),  # always 10
            }
        )

    group_summary_df = pd.DataFrame(group_summary)

    # QA check
    assert (group_summary_df["n_obs"] == 10).all()

    # Save group summary CSV to buffer
    summary_csv_buffer = BytesIO()
    group_summary_df.to_csv(summary_csv_buffer, index=False)
    summary_csv_buffer.seek(0)

    # Create KDE plot
    png_buffer = BytesIO()
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.kdeplot(
        x=group_summary_df["group_mean"],
        weights=group_summary_df["n_obs"],
        fill=True,
        alpha=0.5,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(png_buffer, format="png", dpi=200)
    plt.close(fig)

    png_buffer.seek(0)

    return summary_csv_buffer.getvalue(), png_buffer.getvalue()


# ---------------------------------------------------------------------
# Lab-specific wrappers
# ---------------------------------------------------------------------
def process_alp(df, value_col):
    alp_codes = ["109532-2", "1783-0", "59164-4", "16337-8"]
    filtered_df = filter_lab_data(df, alp_codes, value_col)

    return generate_grouped_mean_density(
        filtered_df,
        value_col,
        title="ALP Grouped Mean Density (chunks of 10)",
        xlabel="Mean ALP value (per 10 observations)",
    )


def process_ldl(df, value_col):
    ldl_codes = ["13457-7", "53133-5", "96258-9", "69419-0"]
    filtered_df = filter_lab_data(df, ldl_codes, value_col)

    return generate_grouped_mean_density(
        filtered_df,
        value_col,
        title="LDL Grouped Mean Density (chunks of 10)",
        xlabel="Mean LDL value (per 10 observations)",
    )


# ---------------------------------------------------------------------
# ZIP output
# ---------------------------------------------------------------------
def create_zip_output(df, zip_path, value_col="Observation.value"):
    """
    Create ZIP file containing lab density plots and group summaries
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:

        # ALP
        alp_summary_csv, alp_png_bytes = process_alp(df, value_col)
        if alp_summary_csv is not None:
            zf.writestr("alp_group_summary.csv", alp_summary_csv)
            zf.writestr("alp_density.png", alp_png_bytes)

        # LDL
        ldl_summary_csv, ldl_png_bytes = process_ldl(df, value_col)
        if ldl_summary_csv is not None:
            zf.writestr("ldl_group_summary.csv", ldl_summary_csv)
            zf.writestr("ldl_density.png", ldl_png_bytes)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    input_path = "synth_dataset.csv"
    zip_path = "lab_density_outputs.zip"

    df = load_data(input_path)
    create_zip_output(df, zip_path)


if __name__ == "__main__":
    main()
