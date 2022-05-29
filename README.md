# MLOps Zoomcamp

In this repository, I will put all the exercises for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp).


Requirements are on `requirements.txt` file.

# 01 - Intro
For the intro we will be using the [NYC taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

# 02 - Experiment Tracking
For the intro we will be using the [NYC taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) for Green Taxi trip, from January through March 2021.

You can run the preprocessing using:

```
bash preprocess_data.sh
```

You can start the mlflow server using:

```
bash launch_mlflow_server.sh
```

After the data is preprocessed and the mlflow server is running you only need to run:

```
bash train_and_register.sh
```
