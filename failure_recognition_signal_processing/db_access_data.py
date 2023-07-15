"""Module for loading timeseries from the database"""

import sqlalchemy as db
import yaml
import pandas as pd
from failure_recognition_signal_processing.db_schema import (
    SchemaInterface,
    timeseries_zdg,
)


def load_db_data(table: SchemaInterface, save_data: bool) -> pd.DataFrame:
    """Load a pandas dataframe from the database using the
    data base info provided in db_info.yaml
    """

    with open("examples/db_info.yaml", "r", encoding="utf-8") as user_data_stream:
        db_info = yaml.safe_load(user_data_stream)

    username = db_info["username"]
    password = db_info["userpassword"]
    engine = db.create_engine(
        f'mysql://{username}:{password}@{db_info["host"]}/{db_info["schema"]}'
    )
    connection = engine.connect()
    metadata = db.MetaData()
    metadata.reflect(engine)

    table_name = table.__tablename__
    series_data_frame = pd.read_sql_table(table_name, connection, db_info["schema"])

    series_data_frame.drop(columns=table.get_drop_list(), inplace=True)
    series_data_frame.rename(columns=table.get_rename_dict(), inplace=True)

    if save_data:
        series_data_frame.to_csv(table_name + ".csv")
        series_data_frame.to_pickle(table_name + ".pkl")

    return series_data_frame


if __name__ == "__main__":
    load_db_data(timeseries_zdg, save_data=True)
