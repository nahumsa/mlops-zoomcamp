import pickle
import pandas as pd

from typing import Any, List, Tuple, Protocol

from datetime import datetime
from dateutil.relativedelta import relativedelta


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect.logging import get_run_logger

class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...


class PreprocessingMethod(Protocol):
    def fit(self, X): ...
    def fit_transform(self, X): ...
    def transform(self, X): ...


def save_pickle(save_path: str, object: Any) -> None:
    """Save an object to the save_path

    Args:
        save_path (str): Path to save the object
        object (Any): object that you want to save
    """
    with open(save_path, 'wb') as file:
        pickle.dump(object, file)

@task
def read_data(path: str) -> pd.DataFrame:
    """Reads parquet data for a given path

    Args:
        path (str): path to read

    Returns:
        pd.DataFrame: parquet data
    """
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df: pd.DataFrame, categorical: List[str], train: bool=True) -> pd.DataFrame:
    """ Preprocess the features by creating duration and filtering between 1 and 10 minutes

    Args:
        df (pd.DataFrame): dataframe to preprocess
        categorical (List[str]): categorical variables
        train (bool, optional): True if you are preprocessing the train set. Defaults to True.

    Returns:
        pd.DataFrame: preprocessed data
    """
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
def train_model(df: pd.DataFrame, categorical: List[str], date: str) -> Tuple[PreprocessingMethod, ScikitModel]:
    """ Create a dict vectorizer and train the Linear Regression model.

    Args:
        df (pd.DataFrame): train data
        categorical (List[str]): categorical variables to select from the data
        date (str): data that you want to train

    Returns:
        PreprocessingMethod: DictVectorizer method that was fitted to the train data
        ScikitModel: LinearRegression model that was fitted on the train data
    """
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
def run_model(df: pd.DataFrame, categorical: List[str], dv: PreprocessingMethod, lr: ScikitModel):
    """ Run the model on a given dataframe

    Args:
        df (pd.DataFrame): data
        categorical (List[str]): categorical variables that were used on the train step
        dv (PreprocessingMethod): Method used to preprocess categorical variables on the train step
        lr (ScikitModel): model used on the train step
    """
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date: str) -> Tuple[str, str]:
    """ Create paths for a given date

    Args:
        date (str): date in the format "%Y-%m-%d

    Returns:
        str: path to the train data
        str: path to the test data
    """
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
def main(date: str=None):
    
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