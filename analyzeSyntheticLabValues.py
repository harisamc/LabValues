import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

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

def generate_density_plot(df, value_col, breaks, title):
    '''
    Generate density plot and aggregated counts from binned lab values
    Returns CSV bytes and PNG bytes for in-memory storage
    '''
    # Bin and aggregate
    binned = pd.cut(df[value_col], bins=breaks, include_lowest=True, right=True)
    counts = df.groupby(binned, observed=True).size().reset_index(name="n")
    counts["midpoint"] = counts[binned.name].apply(lambda x: x.left + (x.right - x.left)/2)
    
    # Expand for kernel density estimation
    expanded = counts.loc[counts.index.repeat(counts["n"])].copy()
    
    # CSV in memory
    csv_buffer = BytesIO()
    counts.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # PNG in memory
    png_buffer = BytesIO()
    if not expanded.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(
            expanded["midpoint"].astype(float), 
            fill=True, 
            alpha=0.5,
            clip=(expanded["midpoint"].min(), expanded["midpoint"].max()), 
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel(value_col)
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(png_buffer, format="png", dpi=200)
        plt.close(fig)
        png_buffer.seek(0)
    else:
        print(f"No data to plot for {title}. Skipping plot.")
    
    return csv_buffer.getvalue(), png_buffer.getvalue()

def process_alp(df, value_col):
    '''
    Process ALP lab values: filter, bin, and generate density plot
    '''
    alp_codes = ["109532-2", "1783-0", "59164-4", "16337-8"]
    alp_breaks = np.arange(1, 526, 25) # Adjust
    
    filtered_df = filter_lab_data(df, alp_codes, value_col)
    return generate_density_plot(filtered_df, value_col, alp_breaks, "ALP Density (synth)")

def process_ldl(df, value_col):
    '''
    Process LDL lab values: filter, bin, and generate density plot
    '''
    ldl_codes = ["13457-7", "53133-5", "96258-9", "69419-0"]
    ldl_breaks = np.arange(0.1, 16, 1.5) # Adjust
    
    filtered_df = filter_lab_data(df, ldl_codes, value_col)
    return generate_density_plot(filtered_df, value_col, ldl_breaks, "LDL Density (synth)")

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
    zip_path = "lab_density_outputs.zip"  # Add this line
    df = load_data(input_path)
    create_zip_output(df,zip_path)

if __name__ == "__main__":
    main()
