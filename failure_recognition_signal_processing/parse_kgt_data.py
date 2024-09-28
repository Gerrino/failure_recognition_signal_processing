from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from scipy import stats
from sklearn.metrics import f1_score
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


def extract_sensor_dfs_from_raw_data(machine_df: pd.DataFrame, target_sensors: list) -> List[SensorData]:
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
        if sensor_name.lower() not in [s.lower() for s in target_sensors]:
            continue

        combined_sensor_df = combine_sensor_columns(columns)
        sensor_list.append(SensorData(sensor_name, combined_sensor_df))

    return sensor_list


def create_machine_object(
    machine_file: str, label_file: str, time_range: Tuple[float, float], target_sensors: list, replacement_dates: dict, discarded_timeseries: dict
) -> Machine:
    """Create a machine object for a specific machine file within a given time range."""
    machine_name = Path(machine_file).stem
    replacement_date = replacement_dates.get(machine_name)

    if machine_name in discarded_timeseries:
        discarded_ts_sensors, discarded_ts = discarded_timeseries.get(machine_name)

    machine_df = load_csv_file(machine_file)
    label_df = np.array(pd.read_excel(label_file, header=None).squeeze())

    # Segment the machine data by the provided time range
    machine_df = segment_by_time_range(machine_df, time_range).dropna(axis=1)
    sensor_list = extract_sensor_dfs_from_raw_data(machine_df, target_sensors)
 
    # manually remove some time series
    if machine_name in discarded_timeseries:
        discarded_ts.sort(key=lambda x: -x)
        for sensor in sensor_list:            
            if sensor.sensor_name.lower() in [s.lower() for s in discarded_ts_sensors]:
                for column in discarded_ts:
                    sensor.dataframe = sensor.dataframe.drop(sensor.dataframe.columns[column], axis=1)

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

        for sensor in sensor_list_tmp:
            assert sensor.dataframe.shape[0] == sensor_list_tmp[-1].dataframe.shape[0]
            assert (
                sensor.dataframe.shape[1]
                == sensor_list[-1].dataframe.shape[1] + sensor_list_post_fix[-1].dataframe.shape[1]
            )

        label_df_post_fix = label_df[replacement_date:]
        label_df = label_df[0:replacement_date]

        assert int(label_df_post_fix[0]) == 0
        assert int(label_df[-1]) == 1

    machine = Machine(machine_name, label_df, sensor_list, label_df_post_fix, sensor_list_post_fix)

    return machine


def combine_multi_machine_data(sensor_name: str, training_machines: List[Machine]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge sensor data and labels from all training machines"""

    machine_names = ", ".join([m.machine_name for m in training_machines])
    print(f"Merging training data from machines {machine_names}")

    # merge training labels
    training_labels = training_machines[0].labels
    for machine in training_machines[1:]:
        training_labels = np.concatenate([training_labels, machine.labels], axis=0)

    assert len(training_labels.shape) == 1

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

def machine_sensors_to_tsfresh(machine: Machine) -> Tuple[list, Dict[int, any]]:
    """Convert time, sensor0/datetime0, .., sensor0/datetimeN; ..,sensorM/dateTimeN => id, time, sensor0, .., sensorM
    
    """    
    output_df = pd.DataFrame(columns=["time","id"])
    output_df.set_index("id")

    current_id_map = {}
    current_id = -1

    sensor_df_map = {}

    for sensor in machine.sensors:        
        sensor_number = sensor.sensor_name[1:]

        for i, col_name in enumerate(sensor.dataframe.columns):
            datetime = extract_date_time(col_name)

            if sensor_number not in sensor_df_map:
                sensor_df_map[sensor_number] = pd.DataFrame(columns=["time","id", sensor_number]) 

            if datetime not in current_id_map:
                current_id += 1
                current_id_map[datetime] = current_id
            else:
                current_id = current_id_map.get(datetime)        

            column = pd.Series(sensor.dataframe.iloc[:, i])
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


def example_create_feature_states():
    target_sensors = ["s5", "s9", "s11", "s13", "s15"]
    replacement_dates = {} #{"7": 115, "11": 100, "12": 80, "14": 57} # maps machine number to first index after replacement
    discarded_timeseries = {"2": (["s5", "s15"], list(range(50, 53))), "9": (target_sensors, list(range(39))), "12": (["s5", "s9", "s11", "s13"], [109])}
    time_range = (2, 7.10)  # Example time range from 0 to 120 seconds

    machine_cnt = 15
    target_machines = list(str(x) for x in range(1, machine_cnt))
    target_machines.remove("13")

    machine_list = []

    for i in target_machines:
        machine_csv = Path(f"examples/dumps_kgt_bueh/data_2/train/{i}.csv")
        label_csv = Path(f"examples/dumps_kgt_bueh/data_2/label_segmented/{i}.xlsx")
        sensor_df = pd.read_csv(machine_csv, sep=";", index_col="Time [s]")
        
        try:
            machine = create_machine_object(machine_csv, label_csv, time_range, target_sensors, replacement_dates, discarded_timeseries)
        except Exception as e:
            print(str(e))
            print("Skipping", str(i))
            continue
        else:
            machine_list.append(machine)
        
        machine_tsfresh = machine_sensors_to_tsfresh(machine)

        container = FeatureContainer()
        container.load(PATH_DICT["features"], PATH_DICT["forest_params"])

        container.compute_feature_state(machine_tsfresh, compute_for_all_features=True)
        container.feature_state.to_pickle(f"./examples/dumps_kgt_bueh/data_2/pickled_feature_state_{i}.pkl")


def feature_fixed_threshold_predictor(feature_vector: pd.DataFrame, label_df: pd.DataFrame, feature_threshold: float, type_lower_thres: bool = False): 

    if type_lower_thres:
        predictions = feature_vector < feature_threshold
    else:
        predictions = feature_vector > feature_threshold
    
    false_positives = int(((label_df == False) & (predictions == True)).sum())
    true_negatives = int(((label_df == False) & (predictions == False)).sum())
    false_negatives = int(((label_df == True) & (predictions == False)).sum())
    true_positives = int(((label_df == True) & (predictions == True)).sum())
    
    if false_positives + true_negatives > 0:
        false_positive_rate =  false_positives / (false_positives + true_negatives)
    else:
        false_positive_rate = 0
   
    f1 = f1_score(label_df, predictions)
    
    return f1, false_positive_rate


if __name__ == "__main__":
    #example_create_feature_states()

    
    target_sensors = ["s5", "s9", "s11", "s13", "s15"]
    replacement_dates = {"7": 115, "11": 100, "12": 80, "14": 57} # maps machine number to first index after replacement

    machine_cnt = 15
    target_machines = list(str(x) for x in range(1, machine_cnt))
    target_machines.remove("13")

    machine_labels: Dict[str, np.ndarray] = {}
    machine_features: Dict[str, pd.DataFrame] = {}

    machine_labels_post_fix: Dict[str, np.ndarray] = {}
    machine_features_post_fix: Dict[str, pd.DataFrame] = {}

    for m in target_machines:
        label_file = Path(f"examples/dumps_kgt_bueh/data_2/label_segmented/{m}.xlsx")
        label_df = np.array(pd.read_excel(label_file, header=None).squeeze())
        
        feature_state = pd.read_pickle(f"./examples/dumps_kgt_bueh/data_2/pickled_feature_state_{m}.pkl")

        if m in replacement_dates:
            feature_state_post_fix = feature_state.iloc[replacement_dates.get(m):, :]
            feature_state = feature_state.iloc[0:replacement_dates.get(m), :]
            label_df_post_fix = label_df[replacement_dates.get(m):]
            label_df = label_df[0:replacement_dates.get(m)]
            machine_features_post_fix[m] = feature_state_post_fix
            machine_labels_post_fix[m] = label_df_post_fix
            
        machine_features[m] = feature_state
        machine_labels[m] = label_df                

    machine_f1_scores = []
    machine_fpr = []
    machine_f1_scores_post_fix = []
    machine_fpr_post_fix = []

    # Cross correlation
    for test_machine in target_machines:
        # Combine feature training data
        combined_features = None
        combined_labels = None
        training_machines = [mj for mj in target_machines if mj != m]

        assert len(training_machines) == len(target_machines) - 1
       
        for mj in training_machines:
            if combined_features is None:
                combined_features = machine_features[mj]
                combined_labels = machine_labels[mj]
            else:
                combined_features = pd.concat([combined_features, machine_features[mj]], axis=0)
                combined_labels = np.concatenate([combined_labels, machine_labels[mj]], axis=0)

        # Find highest ranked feature
        spearman_coeffs = []
        min_corr, max_corr = (-1, 1), (-1, 0)
        for i in range(combined_features.shape[1]):
            feature = combined_features.iloc[:, i]
            corr, _ = stats.spearmanr(feature, combined_labels)
            spearman_coeffs.append(corr)
            if corr < min_corr[1]:
                min_corr = i, corr
            if corr > max_corr[1]:
                max_corr = i, corr
        
        print(f"Found min corr {combined_features.columns[min_corr[0]]}: {min_corr[1]} and max corr {combined_features.columns[max_corr[0]]}: {max_corr[1]}")

        use_min_corr = False
        feature_index = max_corr[0]
        if -min_corr[1] > max_corr[1]:
            use_min_corr = True
            feature_index = min_corr[0]

        opt_feature_vector = np.array(combined_features.iloc[:, feature_index].squeeze())

        # Optimize feature treshold
        opt_feature_vector_1 = opt_feature_vector[combined_labels == True]
        opt_feature_vector_0 = opt_feature_vector[combined_labels == False]

        opt_threshold = opt_feature_vector_0.mean()
        opt_f1 = 0

        opt_steps = 100
        for i in range(opt_steps):
            threshold = opt_threshold + (opt_feature_vector_1.mean() - opt_feature_vector_0.mean())/opt_steps*i
            f1, fpr = feature_fixed_threshold_predictor(opt_feature_vector, combined_labels, threshold, type_lower_thres=use_min_corr)
            
            if f1 > opt_f1:
                opt_f1 = f1
                opt_threshold = threshold

        # Test predictor
        test_feature_vector = np.array(machine_features[test_machine].iloc[:, feature_index].squeeze())
        test_labels = machine_labels[test_machine]

        f1, fpr = feature_fixed_threshold_predictor(test_feature_vector, test_labels, opt_threshold, type_lower_thres=use_min_corr)        
       
        machine_f1_scores.append(f1)
        machine_fpr.append(fpr)

        if test_machine in machine_features_post_fix:
            test_feature_vector_post_fix = np.array(machine_features_post_fix[test_machine].iloc[:, feature_index].squeeze())
            test_labels_post_fix = machine_labels_post_fix[test_machine]

            f1_post_fix, fpr_post_fix = feature_fixed_threshold_predictor(test_feature_vector_post_fix, test_labels_post_fix, opt_threshold, type_lower_thres=use_min_corr)

            machine_f1_scores_post_fix.append(f1_post_fix)
            machine_fpr_post_fix.append(fpr_post_fix)
        else:
            machine_f1_scores_post_fix.append(None)
            machine_fpr_post_fix.append(None)
    pass