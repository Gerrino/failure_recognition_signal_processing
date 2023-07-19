"""Analyse error classes for a sensor in time domain"""
from typing import Dict, List
import pandas as pd


class TimeDomainErrorAnalysis:
    """Class for storing timeseries for error analysus"""

    timeseries_error_negative: pd.DataFrame  # error=0 (target_error=0)
    timeseries_target_error: pd.DataFrame  # target_error=1 (error=1)
    timeseries_other_error: pd.DataFrame  # error=1, target_error=0


def convert_from_tsfresh(df_tsfresh: pd.DataFrame, sensor: int, group_by: str = "id"):
    """Given a dataframe in tsfresh format (row: (time, id, sensors)) extract
    a dataframe for a single sensor (size: N_TIME x N_TIMESERIES, row: (sensor_Tj_ti))
    """
    if any((missing := x) not in df_tsfresh for x in [group_by, "time", sensor]):
        raise ValueError(f"convert_from_tsfresh: Missing columns {missing}!")

    error_negative_list = list(df_tsfresh.groupby(group_by))

    timeseries_list: List[pd.DataFrame] = [
        x[1][["time", sensor]].set_index("time") for x in error_negative_list
    ]
    joined_timeseries = pd.concat(timeseries_list, axis=1)

    return joined_timeseries


def extract_compare_time_domain(
    classification: pd.DataFrame,
    df_timeseries: pd.DataFrame,
    sensor: str,
    target_error_class: str,
    error_class: str = "fehler",
):
    """Extract time series for a specific error case and no error

    Paramaters
    ---
    classification: pd.DataFrame, classification matrix
    df_timeseries: pd.DataFrame, tsfresh format,
    sensor: str, name of the target sensor column
    target_error_class: str, name of the target error column to analyse
    error_class: str, name of the classification column indicating error/no-error

    Returns
    ---
    error_analysis: TimeDomainErrorAnalysis
    """
    if any(
        (missing := x) not in classification for x in [error_class, target_error_class]
    ):
        raise ValueError(
            f"Error column '{missing}' not found in classification matrix!"
        )

    if sensor not in df_timeseries:
        raise ValueError(f"Sensor column '{sensor}' not found in timeseries matrix!")

    # classification_index_to_timeseries_id <=> classification.Zyklus_Nummer
    classification = classification[[error_class, target_error_class, "Zyklus_Nummer"]]
    df_timeseries = df_timeseries[["time", "id", sensor]]

    no_error_rows = classification[classification[error_class] == 0]
    target_error_rows = classification[classification[target_error_class] == 1]
    error_other_error = classification[
        (classification[error_class] == 1) & (classification[target_error_class] == 0)
    ]

    joined_error_negative = df_timeseries.join(
        no_error_rows.set_index("Zyklus_Nummer"), "id", how="right"
    )
    # same as no_error_rows.join(df_timeseries.set_index("id"), "Zyklus_Nummer") # how="left"
    # same as: no_error_rows.merge(df_timeseries, left_on="Zyklus_Nummer", right_on="id")

    joined_target_error = df_timeseries.join(
        target_error_rows.set_index("Zyklus_Nummer"), "id", how="right"
    )
    joined_other_error = df_timeseries.join(
        error_other_error.set_index("Zyklus_Nummer"), "id", how="right"
    )

    analysis = TimeDomainErrorAnalysis()
    analysis.timeseries_error_negative = convert_from_tsfresh(
        joined_error_negative, sensor
    )
    analysis.timeseries_target_error = convert_from_tsfresh(joined_target_error, sensor)
    analysis.timeseries_other_error = convert_from_tsfresh(joined_other_error, sensor)

    n_error_pos = (
        analysis.timeseries_target_error.shape[1]
        + analysis.timeseries_other_error.shape[1]
    )
    n_error_neg = analysis.timeseries_error_negative.shape[1]

    if n_error_pos + n_error_neg != len(set(classification.Zyklus_Nummer)):
        raise ValueError(f"Classification {error_class} may only have values [0,1]")

    return analysis


if __name__ == "__main__":
    classification: pd.DataFrame = pd.read_csv(
        "./examples/dumps/bue_data_classification_20230601.tsv", delimiter="\t"
    )

    df_timeseries: pd.DataFrame = pd.read_pickle("./examples/dumps/timeseries_zdg.pkl")

    analysis = extract_compare_time_domain(
        classification, df_timeseries, "p_Cav1", "fehler_aufschweissung"
    )

    print(analysis)
