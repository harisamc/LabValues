import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

'''
The script processes a data frame of lab values and results in a density plot of the lab values. 
For every 10 rows in the data set of the respective lab value, compute a mean value which will be 
be used for the plotting. 
Output: density plots and csv files of the exact values, i.e. means of chunks of 10
'''

def load_data(filepath):
    '''
    Load the synthetic dataset with date parsing for relevant columns
    '''
    df = pd.read_csv(filepath, parse_dates=[
        "Patient.birthDate", 
        "Condition.recordedDate", 
        "Observation.effective.x.extension.QuelleKlinischesBezugsdatum"
    ])
    return df

def filter_lab_data(df, loinc_codes, value_col):
    '''
    Filter dataframe for specific LOINC codes and keep only positive values
    '''
    filtered = df[df["Observation.code"].isin(loinc_codes)].copy()
    filtered = filtered[filtered[value_col] > 0]
    return filtered

def generate_row_group_density(df, value_col, title, xlabel,rows_per_group=10):
    """
    Take the dataframe as is, without grouping by patient
    Walk through it in chunks of 10 rows and compute the mean per chunk.
    """
    # Ensure consistent order
    df = df.sort_index().reset_index(drop=True)

    # Prepare storage
    group_means = []

    # Loop through in steps of roww_per_group
    for start in range(0, len(df), rows_per_group):
        end = start + rows_per_group
        chunk = df.iloc[start:end]

        if len(chunk) == 0:
            continue

        mean_val = chunk[value_col].mean()
        group_means.append(mean_val)

    # Convert to data frame
    grouped = pd.DataFrame({
        "group_index": range(len(group_means)),
        "group_mean": group_means
    })

    csv_buffer = BytesIO()
    grouped.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)


    png_buffer = BytesIO()

    if not grouped.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(grouped["group_mean"], fill=True, alpha=0.5, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(png_buffer, format="png", dpi=200)
        plt.close(fig)
        png_buffer.seek(0)

    return csv_buffer.getvalue(), png_buffer.getvalue()



def process_alp(df, value_col):
    '''
    Process ALP lab values: filter, bin, and generate density plot
    '''
    alp_codes = ["109532-2", "1783-0", "59164-4", "16337-8"]    
    filtered_df = filter_lab_data(df, alp_codes, value_col)
    return generate_row_group_density(filtered_df, value_col,
    xlabel = "Mean ALP value (mmol/L, per 10 rows)",
    title = "ALP Grouped Mean Density")


def process_ldl(df, value_col):
    '''
    Process LDL lab values: filter, bin, and generate density plot
    '''
    ldl_codes = ["13457-7", "53133-5", "96258-9", "69419-0"]    
    filtered_df = filter_lab_data(df, ldl_codes, value_col)
    return generate_row_group_density(filtered_df, value_col, 
    xlabel = "Mean LDL value (U/L, per 10 rows)", 
    title = "LDL Grouped Mean Density")

def create_zip_output(df, zip_path, value_col="Observation.value"):
    '''
    Create ZIP file containing all lab density plots and counts
    '''
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # ALP
        alp_csv_bytes, alp_png_bytes = process_alp(df, value_col)
        zf.writestr("alp_counts_synth.csv", alp_csv_bytes)
        zf.writestr("alp_density_synth.png", alp_png_bytes)
        
        # LDL
        ldl_csv_bytes, ldl_png_bytes = process_ldl(df, value_col)
        zf.writestr("ldl_counts_synth.csv", ldl_csv_bytes)
        zf.writestr("ldl_density_synth.png", ldl_png_bytes)
    

def main():
    input_path = "synth_dataset.csv" 
    zip_path = "lab_density_outputs.zip"  
    df = load_data(input_path)
    create_zip_output(df,zip_path)

if __name__ == "__main__":
    main()
