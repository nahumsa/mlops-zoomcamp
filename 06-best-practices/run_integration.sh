#!/usr/bin/env bash
export PIPENV_VERBOSITY=-1
export INPUT_FILE_PATTERN="s3://nyc-duration/taxi_type=fhv/year=2021/month=01/predictions.parquet"

docker-compose up -d
sleep 1
aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration

pipenv run python tests/integration_test.py
pipenv run python batch.py --month 1 --year 2021

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi


docker-compose down