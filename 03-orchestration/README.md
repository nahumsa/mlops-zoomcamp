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

What is the file size of the DictVectorizer that we trained when the date is 2021-08-15?

- [X] 13,000 bytes
- [ ] 23,000 bytes
- [ ] 33,000 bytes
- [ ] 43,000 bytes


# Q4. Creating a deployment with a CronSchedule
What is the Cron expression to run a flow at 9 AM every 15th of the month?

- [ ] * * 15 9 0
- [ ] 9 15 * * *
- [X] 0 9 15 * *
- [ ] 0 15 9 1 *


# Q5. Viewing the Deployment

How many flow runs are scheduled by Prefect in advance? You should not be counting manually. There is a number of upcoming runs on the top right of the dashboard.

- [ ] 0
- [X] 3
- [ ] 10
- [ ] 25

# Q6. Creating a work-queue
What is the command to view the available work-queues?

- [ ] prefect work-queue inspect
- [X] prefect work-queue ls
- [ ] prefect work-queue preview
- [ ] prefect work-queue list