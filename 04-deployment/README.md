Exercise

# Q1. Notebook
We'll start with the same notebook we ended up with in homework 1.

We cleaned it a little bit and kept only the scoring part. Now it's in homework/starter.ipynb.

Run this notebook for the February 2021 FVH data.

What's the mean predicted duration for this dataset?

- [ ] 11.19
- [X] 16.19
- [ ] 21.19
- [ ] 26.19

# Q2. Preparing the output
Like in the course videos, we want to prepare the dataframe with the output.

First, let's create an artificial ride_id column:
```python
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
```

Next, write the ride id and the predictions to a dataframe with results.

Save it as parquet:

```python
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)
```

What's the size of the output file?

- [ ] 9M
- [X] 19M
- [ ] 29M
- [ ] 39M

# Q3. Creating the scoring script
Now let's turn the notebook into a script.

Which command you need to execute for that?

`jupyter nbconvert --to=script starter.ipynb`

# 