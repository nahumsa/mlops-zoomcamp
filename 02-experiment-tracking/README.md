# Homework

## Q1. Install MLflow

What's the version that you have? 1.26.0


## Q2. Download and preprocess the data

How many files were saved to OUTPUT_FOLDER? 4.

`dv.pkl`, `test.pkl`, `train.pkl`, `valid.pkl`.

## Q3. Train a model with autolog

How many parameters are automatically logged by MLflow? 17.

- bootstrap:	True
- ccp_alpha:	0.0
- criterion:	squared_error
- max_depth:	10
- max_features:	1.0
- max_leaf_nodes:	None
- max_samples:	None
- min_impurity_decrease:	0.0
- min_samples_leaf:	1
- min_samples_split:	2
- min_weight_fraction_leaf:	0.0
- n_estimators:	100
- n_jobs:	None
- oob_score:	False
- random_state:	0
- verbose:	0
- warm_start:	False

## Q4. Launch the tracking server locally

In addition to backend-store-uri, what else do you need to pass to properly configure the server?

- [X] default-artifact-root
- [ ] serve-artifacts
- [ ] artifacts-only
- [ ] artifacts-destination

## Q5. Tune the hyperparameters of the model

What's the best validation RMSE that you got?

- [ ] 6.128
- [X] 6.628
- [ ] 7.128
- [ ] 7.628

## Q6. Promote the best model to the model registry
What is the test RMSE of the best model?

- [ ] 6.1
- [X] 6.55
- [ ] 7.93
- [ ] 15.1

