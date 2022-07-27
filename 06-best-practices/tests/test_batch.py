import pandas as pd

from datetime import datetime

from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

def create_data() -> pd.DataFrame:
    data = [
        (None, None, dt(hour=1, minute=2), dt(hour=1, minute=10)),
        (1, 1, dt(hour=1, minute=2), dt(hour=1, minute=10)),
        (1, 1, dt(hour=1, minute=2, second=0), dt(hour=1, minute=2, second=50)),
        (1, 1, dt(hour=1, minute=2, second=0), dt(hour=2, minute=2, second=1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)
    return df

def expected_data_transform() -> pd.DataFrame:
    data = [
        (-1, -1, 8.0),
        (1, 1, 8.0),
    ]

    columns = ['PUlocationID', 'DOlocationID', 'duration']
    df = pd.DataFrame(data, columns=columns)
    df[['PUlocationID', 'DOlocationID']] = df[['PUlocationID', 'DOlocationID']].astype('str')
    return df
    

def test_prepare_data():
    df = create_data()
    
    categorical = ['PUlocationID', 'DOlocationID']
    
    df = prepare_data(df, categorical=categorical)
    expected_df = expected_data_transform()
    
    pd.testing.assert_frame_equal(df, expected_df)

    