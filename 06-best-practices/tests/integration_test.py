import pandas as pd

from datetime import datetime


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

if __name__ == '__main__':
    S3_ENDPOINT_URL = "http://localhost:4566"
    
    df_input = create_data()
    
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }
    year = 2021
    month = 1
    input_file = f's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    
    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )