import pickle
import pandas as pd

from typing import Any, List

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.logging import get_run_logger

def save_pickle(save_name: str, object: Any):
    with open(save_name, 'wb') as file:
        pickle.dump(object, file)

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df: pd.DataFrame, categorical: List[str], date: str):
    logger = get_run_logger()
    
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    
    save_pickle(f'./models/dv-{date}.pkl', dv)
    save_pickle(f'./models/model-{date}.pkl', lr)
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date: str):
    logger = get_run_logger()
    if isinstance(date, str):
        base_date = datetime.strptime(date, '%Y-%m-%d')
    else:
        base_date = datetime.now()
    train_date = base_date - relativedelta(months=2)
    val_date = base_date - relativedelta(months=1)
    base_path = lambda _date: f'./data/fhv_tripdata_{_date.year}-{_date.strftime("%m")}.parquet'
    train_path, val_path = base_path(train_date), base_path(val_date)
    return train_path, val_path
    

@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical, date).result()
    run_model(df_val_processed, categorical, dv, lr)

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

# main(date="2021-03-10")
# main(date="2021-08-15")

DeploymentSpec(
    flow=main,
    parameters={"date": "2021-08-15"},
    name="nyc-taxi-time-prediction",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=['ml']
)