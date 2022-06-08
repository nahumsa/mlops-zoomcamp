# Q1 - Converting the script to a Prefect flow

We want to bring this to workflow orchestration to add observability around it. The main function will be converted to a flow and the other functions will be tasks. After adding all of the decorators, there is actually one task that you will need to call .result() for inside the flow to get it to work. Which task is this?

- [ ] read_data
- [ ] prepare_features
- [X] train_model
- [ ] run_model

# Q2 - Parameterizing the flow

Download the relevant files needed to run the main flow if date is 2021-08-15. The validation MSE is:

- [X] 11.637
- [ ] 11.837
- [ ] 12.037
- [ ] 12.237

# Q3. Saving the model and artifacts