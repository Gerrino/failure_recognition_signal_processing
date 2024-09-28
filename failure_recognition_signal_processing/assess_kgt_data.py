from typing import Tuple
from matplotlib import pyplot as plt

# rom sklearn.base import accuracy_score
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def split_training_data_by_label(sensor_df: pd.DataFrame, label_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_data_1 = sensor_df.loc[:, label_df == True]
    test_data_0 = sensor_df.loc[:, label_df == False]

    return test_data_0, test_data_1


def assess_test_results_in_envelope(machine_name: str, labels: pd.DataFrame, predictions: pd.DataFrame) -> tuple:

    assert labels.shape == predictions.shape
    assert len(labels.shape) == 1

    false_positives = int(((labels == False) & (predictions == True)).sum())
    true_negatives = int(((labels == False) & (predictions == False)).sum())
    false_negatives = int(((labels == True) & (predictions == False)).sum())
    true_positives = int(((labels == True) & (predictions == True)).sum())

    assert(false_positives + true_negatives + false_negatives + true_positives == len(predictions))

    if false_positives + true_negatives > 0:
        false_positive_rate = (machine_name, false_positives / (false_positives + true_negatives))
    else:
        false_positive_rate = (machine_name, 0)

    if (labels == True).sum() > 0:
        f1 = f1_score(labels, predictions)
        f1_manual = (2*true_positives) / (2*true_positives + false_positives + false_negatives)        

        if abs(f1 - f1_manual) > 0.01:
            raise ValueError("ERROR, F1 error f1, f1_manual", f1, f1_manual)

        test_f1_score = (machine_name, f1)
    else:
        test_f1_score = (machine_name, None)

    return false_positive_rate, test_f1_score


def classify_using_envelope(
    test_df: pd.DataFrame, envelope: Tuple[pd.Series, pd.Series], min_total_violation: int = 100
):
    predictions = []

    env_low = envelope[0].squeeze()
    env_high = envelope[1].squeeze()

    for _, series in test_df.items():
        series = series.squeeze()
        violations = (series < env_low) | (series > env_high)
        predictions.append(violations.sum() > min_total_violation)

    return np.array(predictions)


def calculate_envelope(sensor_df: pd.DataFrame, k: float = 1.0) -> Tuple[pd.Series, pd.Series]:
    # Calculate mean and standard deviation along each column (if multiple columns are present)
    mean_values = sensor_df.mean(axis=1)
    std_values = sensor_df.std(axis=1)

    # Calculate the envelope as mean + k * std
    envelope_high = mean_values + k * std_values
    envelope_low = mean_values - k * std_values

    return envelope_low.squeeze(), envelope_high.squeeze()


def plot_sensor_envelope(
    envelope: tuple,
    sensor_label_0: pd.DataFrame,
    sensor_label_1: pd.DataFrame,
    k,
    title: str,
    sensor_label_0_post_fix: pd.DataFrame = None,
):
    """
    Plot the envelope (mean + k*std) for a given sensor's dataframe.

    Parameters:
    - sensor_df: DataFrame containing sensor data (single sensor with multiple columns)
    """

    # Plot the original data and the envelope
    plt.figure(figsize=(10, 6))

    # Plot each column in the dataframe
    for column in sensor_label_0.columns:
        plt.plot(sensor_label_0.index, sensor_label_0[column], label=None, color="green", alpha=0.7)

    # Plot each column in the dataframe
    for column in sensor_label_1.columns:
        plt.plot(sensor_label_1.index, sensor_label_1[column], label=None, color="red", alpha=0.7)

    if sensor_label_0_post_fix is not None:
        for column in sensor_label_0_post_fix.columns:
            plt.plot(
                sensor_label_0_post_fix.index,
                sensor_label_0_post_fix[column],
                label=None,
                color="orange",
                alpha=0.7,
                linewidth=1.2,
            )

    # Plot the envelope
    plt.plot(sensor_label_0.index, envelope[1], label=f"Envelope High (mean + {k}*std)", color="black", linewidth=1)
    plt.plot(sensor_label_0.index, envelope[0], label=f"Envelope Low (mean - {k}*std)", color="black", linewidth=1)

    # Add titles and labels
    plt.title(f"{title} - Sensor Data with Envelope")
    plt.xlabel("Time [s]")
    plt.ylabel("Sensor Value")
    plt.legend(loc="best")

    # Show the plot
    plt.show()
