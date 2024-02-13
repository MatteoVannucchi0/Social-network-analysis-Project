import pandas as pd


def positive_analysis(df: pd.DataFrame, source, targets=None) -> pd.DataFrame:
    if targets is None:
        output = df[df[source] == source & df["Goldstein"] >= 0]
    else:
        output = df[df[source] == source & df[target].isin(targets) & df["Goldstein"] >= 0]
    return output


def aggregate_by_country(df: pd.DataFrame, aggregators, self_include=True) -> pd.DataFrame:
    df = df[df['Source code'] != df['Target code']]
    return df.groupby(aggregators).agg({'Goldstein': 'sum'}).reset_index().reset_index()
