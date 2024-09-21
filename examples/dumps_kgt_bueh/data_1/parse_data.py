from dataclasses import dataclass, field
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.model_selection import KFold

class SensorData:
    def __init__(self, sensor_name: str, dataframe: pd.DataFrame):
        self.sensor_name = sensor_name
        self.dataframe = dataframe

@dataclass
class Machine:
    machine_name: str    
    labels: pd.DataFrame
    sensors: List[SensorData] = field(default_factory=list)

    def add_sensor(self, sensor_data: SensorData):
        self.sensors.append(sensor_data)

        if sensor_data.dataframe.shape[1] != len(self.labels):
            raise ValueError(f"Error, number of labels ({len(self.labels)}) != sensor data columns ({sensor_data.dataframe.shape[1]})")

    def get_sensor_data_by_name(self, sensor_name: str) -> SensorData:
        sensor_data = [x for x in self.sensors if x.sensor_name.lower() == sensor_name.lower()]
        return sensor_data[0]

def load_csv_file(filepath: str) -> pd.DataFrame:
    """Load a single CSV file into a DataFrame."""
    return pd.read_csv(filepath, sep=';', index_col='Time [s]')

def segment_by_time_range(dataframe: pd.DataFrame, time_range: Tuple[float, float]) -> pd.DataFrame:
    """Segment the data based on the provided time range."""
    start_time, end_time = time_range
    return dataframe[(dataframe.index >= start_time) & (dataframe.index <= end_time)]

def extract_sensor_name(column_name: str) -> str:
    """Extract sensor name from the column name."""
    return column_name.split('_')[3]  # Assuming the sensor name is always in the same position

def combine_sensor_columns(sensor_columns: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple sensor columns into a single dataframe."""
    return pd.concat(sensor_columns, axis=1)

def create_machine_object(machine_file: str, label_file: str, time_range: Tuple[float, float]) -> Machine:
    """Create a machine object for a specific machine file within a given time range."""
    machine_name = os.path.splitext(os.path.basename(machine_file))[0]
    machine_df = load_csv_file(machine_file)
    label_df = pd.read_excel(label_file, header=None)
    
    # Segment the machine data by the provided time range
    segmented_df = segment_by_time_range(machine_df, time_range)

    machine_df = segmented_df.dropna(axis=1)
    
    # Create machine object
    machine = Machine(machine_name, label_df)
    
    # Dictionary to group columns by sensor name
    sensor_columns = {}
    
    # Group columns by sensor name
    for column in machine_df.columns:
        sensor_name = extract_sensor_name(column)
        if sensor_name not in sensor_columns:
            sensor_columns[sensor_name] = []
        sensor_columns[sensor_name].append(machine_df[[column]])  # Append the dataframe for this column
    
    # For each sensor, combine the relevant columns and store them as SensorData
    for sensor_name, columns in sensor_columns.items():
        combined_sensor_df = combine_sensor_columns(columns)
        machine.add_sensor(SensorData(sensor_name, combined_sensor_df))
    
    return machine


def calculate_envelope(sensor_df: pd.DataFrame, k: float = 1.0) -> Tuple[pd.Series, pd.Series]:
     # Calculate mean and standard deviation along each column (if multiple columns are present)
    mean_values = sensor_df.mean(axis=1)
    std_values = sensor_df.std(axis=1)
    
    # Calculate the envelope as mean + k * std
    envelope_high = mean_values + k * std_values
    envelope_low = mean_values - k * std_values

    return envelope_low.squeeze(), envelope_high.squeeze()


def plot_sensor_envelope(envelope: tuple, sensor_label_0: pd.DataFrame, sensor_label_1: pd.DataFrame, k):
    """
    Plot the envelope (mean + k*std) for a given sensor's dataframe.
    
    Parameters:
    - sensor_df: DataFrame containing sensor data (single sensor with multiple columns)
    """
   
    
    # Plot the original data and the envelope
    plt.figure(figsize=(10, 6))
    
    # Plot each column in the dataframe
    for column in sensor_label_0.columns:
        plt.plot(sensor_label_0.index, sensor_label_0[column], label=None, color='green', alpha=0.7)

    # Plot each column in the dataframe
    for column in sensor_label_1.columns:
        plt.plot(sensor_label_1.index, sensor_label_1[column], label=None, color='red', alpha=0.7)
    
    # Plot the envelope
    plt.plot(sensor_label_0.index, envelope[1], label=f'Envelope High (mean + {k}*std)', color='black', linewidth=3)
    plt.plot(sensor_label_0.index, envelope[0], label=f'Envelope Low (mean - {k}*std)', color='black', linewidth=3)

    # Add titles and labels
    plt.title('Sensor Data with Envelope')
    plt.xlabel('Time [s]')
    plt.ylabel('Sensor Value')
    plt.legend(loc='best')
    
    # Show the plot
    plt.show()


# Example usage:
label_file = "examples\dumps_kgt_bueh\data_1\labels\\11.xlsx"
machine_file = "examples\dumps_kgt_bueh\data_1\machine_data\\11.csv"
time_range = (0, 17.70)  # Example time range from 0 to 120 seconds

machine = create_machine_object(machine_file, label_file, time_range)

sensor_sx = machine.get_sensor_data_by_name("s5")

sensor_labels = machine.labels.set_index(sensor_sx.dataframe.columns)
sensor_labels = sensor_labels.squeeze()

sensor_sx_data_label_1 = sensor_sx.dataframe.loc[:, sensor_labels == True]
sensor_sx_data_label_0 = sensor_sx.dataframe.loc[:, sensor_labels == False]

envelope = calculate_envelope(sensor_sx_data_label_0, 2)

plot_sensor_envelope(envelope, sensor_sx_data_label_0, sensor_sx_data_label_1, 2)



pass

