#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import pandas as pd
import argparse

from typing import Union, List, Protocol, Tuple

def read_data(filename: str, categorical: List[str]) -> pd.DataFrame:
    # load the data from S3
    if "s3" in filename.split("://"):
        S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "http://localhost:4566")
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }
        df = pd.read_parquet(filename, storage_options=options)
        
    else:        
        df = pd.read_parquet(filename)
    
    df = prepare_data(df, categorical)
    
    return df

def prepare_data(df: pd.DataFrame, categorical: List[str]) -> pd.DataFrame:
    
    _df = df.copy()
    _df['duration'] = _df.dropOff_datetime - _df.pickup_datetime
    _df['duration'] = _df.duration.dt.total_seconds() / 60

    _df = _df[(_df.duration >= 1) & (_df.duration <= 60)].copy()

    _df[categorical] = _df[categorical].fillna(-1).astype('int').astype('str')
    
    _df = _df.drop(["dropOff_datetime", "pickup_datetime"], axis=1)
    
    return _df

def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--year",
        default=2021,
        help="Year of the data."
    )
    
    parser.add_argument(
        "--month",
        default=2,
        help="Month of the data."
    )
    
    parser.add_argument(
        "--local",
        default=True,
        help="True if saving locally."
    )
    
    return parser.parse_args()

def str2bool(arg_str: Union[str, bool]) -> bool:
    if isinstance(arg_str, bool):
        return arg_str
    if arg_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class DictVectorizer(Protocol):
    def transform(self, X): ...

class ScikitModel(Protocol):
    def fit(self, X, y=None): ...
    def predict(self, X): ...

def load_model(path: str) -> Tuple[DictVectorizer, ScikitModel]:
    with open(path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr

def get_input_path(year, month):
    DEFAULT_INPUT_PATTERN = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', DEFAULT_INPUT_PATTERN)
    print(input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    DEFAULT_OUTPUT_PATTERN = 's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', DEFAULT_OUTPUT_PATTERN)
    return output_pattern.format(year=year, month=month)

def save_prediction(df_result: pd.DataFrame, output_file: str) -> None:
    df_result.to_parquet(output_file, engine='pyarrow', index=False)


def main():
    args = cli()
    year = int(args.year)
    month = int(args.month)
    local = str2bool(args.local)
    input_file = get_input_path(year, month)
    
    if not local:
        output_file = get_output_path(year, month)
        
    elif local:
        output_file = f'taxi_type=fhv_year={year:04d}_month={month:02d}.parquet'

    dv, lr = load_model(path='model.bin')

    categorical = ['PUlocationID', 'DOlocationID']
    
    df = read_data(input_file, categorical=categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())
    print('predicted sum duration:', y_pred.sum())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    save_prediction(df_result, output_file)

if __name__ == '__main__':
    main()