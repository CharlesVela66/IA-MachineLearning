import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from dictionaries import workclass_dict, marital_status_dict, occupation_dict, relationship_dict, race_dict, sex_dict, native_country_dict, income_dict

os.chdir("C:/Users/PC/OneDrive/Documents/UNI/7/Modulo2/dataset")

# Define the original columns of the dataset
cols = ["age", "workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]

# Read the csv
df = pd.read_csv("data_2.csv", names=cols)

# Select the categorical-type feats
categorical_cols = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country","income"]
for col in categorical_cols:
    df[col] = df[col].str.strip()

# Convert categorical columns to integer columns
df['workclass_int'] = df['workclass'].map(workclass_dict)
df['marital_status_int'] = df['marital-status'].map(marital_status_dict)
df['occupation_int'] = df['occupation'].map(occupation_dict)
df['relationship_int'] = df['relationship'].map(relationship_dict)
df['race_int'] = df['race'].map(race_dict)
df['sex_int'] = df['sex'].map(sex_dict)
df['native_country_int'] = df['native-country'].map(native_country_dict)
df['income_int'] = df['income'].map(income_dict)

# Drop the original categorical columns and the ones I will not use
df.drop(['fnlwgt','education','capital-gain','capital-loss','workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income'], axis=1, inplace=True)

# After doing a quick df.info() I saw that there are nearly 2000 columns that have NANS out of 32,000. So I decided to drop these rows even though I will be losing about 1/16 of the data set. I think it's the right choice considering getting the best diagnostics.
# print(df.head())
# print(df.info())

df_cleaned = df.dropna()
df_cleaned.to_csv("cleaned_data.csv", index=False)

print(df_cleaned.head())
print(df_cleaned.info())