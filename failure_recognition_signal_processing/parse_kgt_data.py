from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.model_selection import KFold

from failure_recognition_signal_processing import PATH_DICT
from failure_recognition_signal_processing.feature_container import FeatureContainer


class SensorData:
    def __init__(self, sensor_name: str, dataframe: pd.DataFrame):
        self.sensor_name = sensor_name
        self.dataframe = dataframe


@dataclass
class Machine:
    machine_name: str
    labels: pd.DataFrame
    sensors: List[SensorData] = field(default_factory=list)
    labels_post_fix: pd.DataFrame = None
    sensors_post_fix: List[SensorData] = None

    def add_sensor(self, sensor_data: SensorData):
        self.sensors.append(sensor_data)

        if sensor_data.dataframe.shape[1] != len(self.labels):
            raise ValueError(
                f"Error adding machine {self.machine_name}, number of labels ({len(self.labels)}) != sensor data columns ({sensor_data.dataframe.shape[1]})"
            )

    def get_sensor_data_by_name(self, sensor_name: str, post_fix: bool = False) -> SensorData:
        if post_fix == False:
            sensor_data = [x for x in self.sensors if x.sensor_name.lower() == sensor_name.lower()]
        else:
            sensor_data = [x for x in self.sensors_post_fix if x.sensor_name.lower() == sensor_name.lower()]

        return sensor_data[0]

    def __post_init__(self):
        for sensor in self.sensors:
            if sensor.dataframe.shape[1] != len(self.labels):
                raise ValueError(
                    f"Error adding machine {self.machine_name}, number of labels ({len(self.labels)}) != sensor data columns ({sensor.dataframe.shape[1]})"
                )


def load_csv_file(filepath: str) -> pd.DataFrame:
    """Load a single CSV file into a DataFrame."""
    return pd.read_csv(filepath, sep=";", index_col="Time [s]")


def segment_by_time_range(dataframe: pd.DataFrame, time_range: Tuple[float, float]) -> pd.DataFrame:
    """Segment the data based on the provided time range."""
    start_time, end_time = time_range
    return dataframe[(dataframe.index >= start_time) & (dataframe.index <= end_time)]


def extract_sensor_name(column_name: str) -> str:
    """Extract sensor name from the column name."""
    return column_name.split("_")[3]  # Assuming the sensor name is always in the same position

def extract_date_time(column_name: str) -> datetime:
    """Extract sensor name from the column name."""
    
    _date =  column_name.split("_")[8]  # Assuming the sensor name is always in the same position
    _time =  column_name.split("_")[9]  # Assuming the sensor name is always in the same position

    date_time = datetime.strptime(f"{_date} {_time}", '%Y-%m-%d %H:%M:%S') # 2017-12-05_08:10:05

    return date_time



def combine_sensor_columns(sensor_columns: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple sensor columns into a single dataframe."""
    return pd.concat(sensor_columns, axis=1)


def extract_sensor_dfs_from_raw_data(machine_df: pd.DataFrame) -> List[SensorData]:
    sensor_list: List[SensorData] = []
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
        sensor_list.append(SensorData(sensor_name, combined_sensor_df))

    return sensor_list


def create_machine_object(
    machine_file: str, label_file: str, time_range: Tuple[float, float], replacement_dates: dict
) -> Machine:
    """Create a machine object for a specific machine file within a given time range."""
    machine_name = Path(machine_file).stem
    replacement_date = replacement_dates.get(machine_name)

    machine_df = load_csv_file(machine_file)
    label_df = np.array(pd.read_excel(label_file, header=None))

    # Segment the machine data by the provided time range
    machine_df = segment_by_time_range(machine_df, time_range).dropna(axis=1)
    sensor_list = extract_sensor_dfs_from_raw_data(machine_df)

    # if a tool replacement has occurred split the data
    sensor_list_post_fix = None
    label_df_post_fix = None

    if replacement_date is not None:
        sensor_list_post_fix = []

        sensor_list_tmp = list(sensor_list)
        sensor_list.clear()
        for sensor in sensor_list_tmp:
            sensor_list_post_fix.append(SensorData(sensor.sensor_name, sensor.dataframe.iloc[:, replacement_date:]))
            sensor_list.append(SensorData(sensor.sensor_name, sensor.dataframe.iloc[:, 0:replacement_date]))

            assert sensor.dataframe.shape[0] == sensor_list_tmp[-1].dataframe.shape[0]
            assert (
                sensor.dataframe.shape[1]
                == sensor_list[-1].dataframe.shape[1] + sensor_list_post_fix[-1].dataframe.shape[1]
            )

        label_df_post_fix = label_df[replacement_date:]
        label_df = label_df[0:replacement_date]

        assert int(label_df_post_fix[0][0]) == 0
        assert int(label_df[-1][0]) == 1

    machine = Machine(machine_name, label_df, sensor_list, label_df_post_fix, sensor_list_post_fix)

    return machine


def combine_multi_machine_data(sensor_name: str, training_machines: List[Machine]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge sensor data and labels from all training machines"""

    print(f"Merging training data from {len(training_machines)} machines")

    # merge training labels
    training_labels = training_machines[0].labels
    for machine in training_machines[1:]:
        training_labels = np.concatenate([training_labels, machine.labels], axis=0)

    assert training_labels.shape[1] == 1

    # merge training data series
    training_dfs = [x.get_sensor_data_by_name(sensor_name).dataframe for x in training_machines]
    training_machines[0].labels

    training_df_combined = training_dfs[0]
    for df in training_dfs[1:]:
        training_df_combined = pd.concat([training_df_combined, df], axis=1)

    assert training_labels.shape[0] == training_df_combined.shape[1]

    return (training_df_combined, training_labels)


def convert_to_tsfresh(df_csv: pd.DataFrame, time_range: tuple) -> Tuple[list, Dict[int, any]]:
    """Convert time, sensor0/datetime0, .., sensor0/datetimeN; ..,sensorM/dateTimeN => id, time, sensor0, .., sensorM
    
    """


    time_series_map: Dict[int, list] = {}

    sensor_list = [0, 1, 2, 3, 4, 5, 6, 7]
    
    output_df = pd.DataFrame(columns=["time","id"])
    output_df.set_index("id") 

    df_csv = segment_by_time_range(df_csv, time_range).dropna(axis=1)

    current_id_map = {}
    current_id = -1

    sensor_df_map = {}

    for i, col_name in enumerate(df_csv.columns):

        sensor = extract_sensor_name(col_name)
        datetime = extract_date_time(col_name)
        sensor_number = sensor[1:]

        if sensor_number not in sensor_df_map:
            sensor_df_map[sensor_number] = pd.DataFrame(columns=["time","id", sensor_number]) 

        if datetime not in current_id_map:
            current_id += 1
            current_id_map[datetime] = current_id
        else:
            current_id = current_id_map.get(datetime)        

        column = pd.Series(df_csv.iloc[:, i])
        id_column = pd.Series([current_id] * column.shape[0])
        time_column = pd.Series(column.index)
        column = column.reset_index(drop=True)

        data = {"id": id_column, "time": time_column, sensor_number: column}
        sensor_data_tsfresh = pd.DataFrame(data)
        sensor_data_tsfresh.set_index("id")

        if sensor_df_map[sensor_number].shape[0] == 0:
            sensor_df_map[sensor_number] = sensor_data_tsfresh
        else:
            sensor_df_map[sensor_number] = pd.concat([sensor_df_map[sensor_number], sensor_data_tsfresh], axis=0)

    output_df = sensor_df_map.pop(sensor_number)    
    for sensor, df in sensor_df_map.items():
        output_df = pd.merge(
        left=output_df, 
        right=df,
        how='left',
        left_on=['id', 'time'],
        right_on=['id', 'time'],
        )    
    
    return output_df.dropna(axis=1)

if __name__ == "__main__":
    time_range = (0, 17.70)

    for i in range(1, 16):
        machine_csv = Path(f"examples/dumps_kgt_bueh/data_1/machine_data/{i}.csv")

        if not machine_csv.exists():
            print(f"Skipping machine {machine_csv.stem}")
            continue

        sensor_df = pd.read_csv(machine_csv, sep=";", index_col="Time [s]")

        df_tsfresh = convert_to_tsfresh(sensor_df, time_range)

        container = FeatureContainer()
        container.load(PATH_DICT["features"], PATH_DICT["forest_params"])

        container.compute_feature_state(df_tsfresh, compute_for_all_features=True)
        container.feature_state.to_pickle(f"./examples/dumps_kgt_bueh/data_1/pickled_feature_state_{i}.pkl")

        container.feature_state = pd.read_pickle(f"./examples/dumps_kgt_bueh/data_1/pickled_feature_state_{i}.pkl")

        print(container.feature_state)
        pass