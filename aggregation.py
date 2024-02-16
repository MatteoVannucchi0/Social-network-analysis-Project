import os

import numpy as np
import pandas as pd

from preprocess import preprocessed_path, data_path

VALUE_COUNT_MIN_THRESHOLD: int = 100
PAIR_COUNT_MIN_THRESHOLD: int = 100

aggregated_path = data_path / "aggregated"
aggregated_path.mkdir(exist_ok=True)


# def filter_entities(df: pd.DataFrame) -> pd.DataFrame:
#     # Calculate the value count for each entity
#     source_value_counts = df['Source code'].value_counts()
#     target_value_counts = df['Target code'].value_counts()
#     total_value_counts = source_value_counts.add(target_value_counts, fill_value=0)
#
#     # Filter out entities that have a value count less than the threshold
#     entities_index = total_value_counts[total_value_counts > VALUE_COUNT_MIN_THRESHOLD].index
#     filtered_df = df[(df['Source code'].isin(entities_index)) & (df['Target code'].isin(entities_index))]
#
#     # Print the name of the entities that were kept
#     entities_kept = set(filtered_df['Source code'].unique()) | set(filtered_df['Target code'].unique())
#     starting_entities = set(df['Source code'].unique()) | set(df['Target code'].unique())
#
#     print(f"Entities kept: {', '.join(entities_kept)}")
#     print(f"Entities discarded: {', '.join(starting_entities - entities_kept)}")
#
#     return filtered_df


def aggregate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Filter out entities with a value count less than the threshold
    # df_filtered = filter_entities(df)
    df_filtered = df

    replicated_df = df_filtered.loc[df_filtered.index.repeat(df_filtered["NumEvents"])].reset_index(drop=True).drop(columns=["NumEvents"])

    # take the mean for the column Goldstein
    aggregated_df = replicated_df.groupby(['Source code', 'Target code']).agg({
            'Goldstein': [
                ('sum', 'sum'),
                ('mean', 'mean'),
                ('std', "std"),
                ('count', 'count')
            ]
        }).reset_index()

    aggregated_df.columns = ['_'.join(col).rstrip('_') for col in aggregated_df.columns.values]
    return aggregated_df


def load_dataset(year:int, quantile:float = 0.8) -> pd.DataFrame:
    df = pd.read_csv(aggregated_path / f"aggregated_{year}.csv")
    # Take the pairs in the top 15% percent in terms of number of events
    PAIR_COUNT_MIN_THRESHOLD = df["Goldstein_count"].quantile(quantile, interpolation='higher')
    df = df[df["Goldstein_count"] > PAIR_COUNT_MIN_THRESHOLD]
    return df


def aggregate_data():
    # Preprocess data in data folder
    for file in os.listdir(preprocessed_path):
        year = int(file.split("_")[1].split(".")[0])

        if (aggregated_path / f"aggregated_{year}.csv").exists():
            print(f"File aggregated_{year}.csv already exists, skipping file {file}")
            continue
        try:
            if file.endswith(".csv"):
                df = pd.read_csv(preprocessed_path / file)
                aggregate_df = aggregate_dataframe(df)
                aggregate_df.to_csv(aggregated_path / f"aggregated_{year}.csv", index=False)
        except Exception as e:
            print(f"Error in preprocessing {file} for year {year}: {e}")
            raise e


if __name__ == "__main__":
    aggregate_data()
