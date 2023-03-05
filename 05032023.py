# Databricks notebook source
import pandas as pd

# COMMAND ----------

hausing = pd.read_csv("datasets/housing/housing.csv")

# COMMAND ----------

hausing.head()

# COMMAND ----------

hausing.info()

# COMMAND ----------

hausing["ocean_proximity"].value_counts()

# COMMAND ----------

hausing.describe()

# COMMAND ----------

import matplotlib as plt
hausing.hist(bins=50, figsize=(20, 15))

# COMMAND ----------


