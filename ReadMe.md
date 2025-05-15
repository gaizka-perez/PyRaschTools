# Rasch Py

Rasch Py is a Python library that helps you implement Rasch models in Python. The 'Model' objects can be employed to output the relevant parameters and visualizations. Alternatively, the individual functions can be used on already existing Rasch parameter outputs.

# Code overview

## Import 

```python
from rasch import DichotomousRaschModel #the main model object
from rasch.utils import generate_sample #a sample generator to test our functions
```

## Sample data creation
We can create an example dataset with **generate_sample()**. When used with external data, make sure that it is formatted in long format with a single row per person and a column per item.
```python
num_people = 10  # Number of people (rows)
num_items = 10    # Number of items (columns)
response_options = 2
df = generate_sample(num_people, num_items, response_options)
print(df)
```

## Model implementation and parameter estimation
Once the model object is created, difficulty and ability parameters can be estimated.
```python
model = DichotomousRaschModel(df) #Create the model object
items, people = model.estimate_parameters() #Estimate the model parameters
```

## Rasch plots
After the parameters have been estimated, ICC, IIF, and TIF plots can be generated in a single line of code.
```python
model.ICCplot() #Item characteristic curves
model.IIFplot() #Item information functions
model.TIFplot() #Test information function
```

