import pathlib

import pandas as pd

import pickle
import argparse

from typing import Any, List, Tuple

def load_model(path: str) -> Tuple[Any, Any]:
    with open(path, 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    
    return dv, lr

def read_data(filename: str, features: List[str]) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[features] = df[features].fillna(-1).astype('int').astype('str')
    
    return df

def run(year: int, month: int) -> pd.DataFrame:

    input_file = f'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    categorical = ['PUlocationID', 'DOlocationID']
    model_path = 'model.bin'


    dv, lr = load_model(model_path)
    df = read_data(input_file, features=categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    # Save predictions for the model

    df_prediction = pd.DataFrame()
    df_prediction['ride_id'] = df['ride_id']
    df_prediction['prediction'] = y_pred
    return df_prediction


if __name__ == '__main__':
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
    args = parser.parse_args()
    year = int(args.year)
    month = int(args.month)
    print(year, month)    
    df_prediction = run(year, month)

    mean_prediction = df_prediction['prediction'].mean()
    
    print(f"mean of the prediction for {month:02d}-{year:04d} is {mean_prediction:.02f}")
    
    
    # save predictions for the model to a parquet file
    path = pathlib.Path("output/")
    path.mkdir(parents=True, exist_ok=True)
    output_file = f'output/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    
    df_prediction.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


