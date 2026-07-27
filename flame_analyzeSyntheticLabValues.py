# best final
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
from io import StringIO, BytesIO
import warnings
warnings.filterwarnings('ignore')

from flame.star import StarModel, StarAnalyzer, StarAggregator


class LabDataAnalyzer(StarAnalyzer):
    def __init__(self, flame):
        """
        Initializes the custom Analyzer node.

        :param flame: Instance of FlameCoreSDK to interact with the FLAME components.
        """
        super().__init__(flame)  # Connects this analyzer to the FLAME components

    def filter_lab_data(self, df, loinc_codes, value_col):
        '''
        Filter dataframe for specific LOINC codes and keep only positive values
        '''
        filtered = df[df["Observation.code"].isin(loinc_codes)].copy()
        filtered = filtered[filtered[value_col] > 0]

        return filtered
    
    def process_alp(self, df, value_col):
        '''
        Process ALP lab values: filter, bin, and generate density plot
        '''
        alp_codes = ["109532-2", "1783-0", "59164-4", "16337-8"]
        
        filtered_df = self.filter_lab_data(df, alp_codes, value_col)

        return filtered_df
    
        #print('ALP: ',filtered_df.shape)
        #return generate_density_plot(filtered_df, value_col, alp_breaks, "ALP Density (synth)")
        

    def process_ldl(self, df, value_col):
        '''
        Process LDL lab values: filter, bin, and generate density plot
        '''
        ldl_codes = ["13457-7", "53133-5", "96258-9", "69419-0"]
        
        filtered_df = self.filter_lab_data(df, ldl_codes, value_col)

        return filtered_df
    
        #print('LDL: ',filtered_df.shape)
        #return generate_density_plot(filtered_df, value_col, ldl_breaks, "LDL Density (synth)")
    
    def analysis_method(self, data, aggregator_results):
        """
        Performs analysis on the retrieved data from data sources.

        :param data: A list of dictionaries containing the data from each data source.
                     - Each dictionary corresponds to a data source.
                     - Keys are the queries executed, and values are the results (dict for FHIR, str for S3).

        :return: Any result of your analysis on one node (ex. patient count).
        """
        raw = data[0]['synth_dataset.csv']
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")

        lab_data_df = pd.read_csv(StringIO(raw), parse_dates=[
            "Patient.birthDate",
            "Condition.recordedDate",
            "Observation.effective.x.extension.QuelleKlinischesBezugsdatum"])
        
        alp_df = self.process_alp(lab_data_df, value_col="Observation.value")
        ldl_df = self.process_ldl(lab_data_df, value_col="Observation.value")

        return (alp_df, ldl_df)


class LabDataAggregator(StarAggregator):
    def __init__(self, flame):
        """
        Initializes the custom Aggregator node.

        :param flame: Instance of FlameCoreSDK to interact with the FLAME components.
        """
        super().__init__(flame)  # Connects this aggregator to the FLAME components

        self.zip_path = "lab_density_outputs.zip"

    def generate_row_group_density(self, df, value_col, title, xlabel,rows_per_group=10):
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
    
    
    def create_zip_output(self, alp_df, ldl_df, value_col="Observation.value"):
        '''
        Create ZIP file containing all lab density plots and counts
        '''
        alp_csv_bytes, alp_png_bytes = self.generate_row_group_density(alp_df, value_col,
                                        xlabel = "Mean ALP value (mmol/L, per 10 rows)",
                                        title = "ALP Grouped Mean Density")
        ldl_csv_bytes, ldl_png_bytes = self.generate_row_group_density(ldl_df, value_col,
                                        xlabel = "Mean ALP value (mmol/L, per 10 rows)",
                                        title = "ALP Grouped Mean Density")
        
        with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # ALP write to zip
            zf.writestr("alp_counts_synth.csv", alp_csv_bytes)
            zf.writestr("alp_density_synth.png", alp_png_bytes)
            
            # LDL write to zip
            zf.writestr("ldl_counts_synth.csv", ldl_csv_bytes)
            zf.writestr("ldl_density_synth.png", ldl_png_bytes)


    def aggregation_method(self, analysis_results):
        """
        Aggregates the results received from all analyzer nodes.

        :param analysis_results: A list of analysis results from each analyzer node.
        :return: The aggregated result (e.g., total patient count across all analyzers).
        """

        alp_df, ldl_df = (pd.concat([res[0] for res in analysis_results]),
                          pd.concat([res[1] for res in analysis_results]))
        
        self.create_zip_output(alp_df, ldl_df, value_col="Observation.value")
        
        return open(self.zip_path, 'rb').read()
    


def main():
    """
    Sets up and initiates the distributed analysis using the FLAME components.

    - Defines the custom analyzer and aggregator classes.
    - Specifies the type of data and queries to execute.
    - Configures analysis parameters like iteration behavior and output format.
    """
    StarModel(
        analyzer=LabDataAnalyzer,             # Custom analyzer class (must inherit from StarAnalyzer)
        aggregator=LabDataAggregator,         # Custom aggregator class (must inherit from StarAggregator)
        data_type='s3',                       # Type of data source ('fhir' or 's3')
        query=None,                           # Query or list of queries to retrieve data
        simple_analysis=True,                 # True for single-iteration; False for multi-iterative analysis
        output_type='bytes',                  # Output format for the final result ('str', 'bytes', or 'pickle')
        analyzer_kwargs=None,                 # Additional keyword arguments for the custom analyzer constructor (i.e. MyAnalyzer)
        aggregator_kwargs=None                # Additional keyword arguments for the custom aggregator constructor (i.e. MyAggregator)
    )


if __name__ == "__main__":
    main()
