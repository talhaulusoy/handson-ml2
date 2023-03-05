# Databricks notebook source
# MAGIC %md
# MAGIC **Chapter 2 – End-to-end Machine Learning project**
# MAGIC 
# MAGIC *Welcome to Machine Learning Housing Corp.! Your task is to predict median house values in Californian districts, given a number of features from these districts.*
# MAGIC 
# MAGIC *This notebook contains all the sample code and solutions to the exercices in chapter 2.*

# COMMAND ----------

# MAGIC %md
# MAGIC <table align="left">
# MAGIC   <td>
# MAGIC     <a href="https://colab.research.google.com/github/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# MAGIC   </td>
# MAGIC   <td>
# MAGIC     <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
# MAGIC   </td>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %md
# MAGIC First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20.

# COMMAND ----------

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# COMMAND ----------

# MAGIC %md
# MAGIC # Get the data

# COMMAND ----------

import pandas as pd

housing = pd.read_csv("datasets/housing/housing.csv")

# COMMAND ----------

housing.info()

# COMMAND ----------

housing["ocean_proximity"].value_counts()

# COMMAND ----------

housing.describe()

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import matplotlib.pyplot as plt
# MAGIC housing.hist(bins=30, figsize=(20,15))
# MAGIC save_fig("attribute_histogram_plots")
# MAGIC plt.show()

# COMMAND ----------

# to make this notebook's output identical at every run
np.random.seed(42)

# COMMAND ----------

import numpy as np

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# COMMAND ----------

train_set, test_set = split_train_test(housing, 0.2)
len(train_set)

# COMMAND ----------

len(test_set)

# COMMAND ----------

from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# COMMAND ----------

# MAGIC %md
# MAGIC The implementation of `test_set_check()` above works fine in both Python 2 and Python 3. In earlier releases, the following implementation was proposed, which supported any hash function, but was much slower and did not support Python 2:

# COMMAND ----------

import hashlib

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

# COMMAND ----------

# MAGIC %md
# MAGIC If you want an implementation that supports any hash function and is compatible with both Python 2 and Python 3, here is one:

# COMMAND ----------

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio

# COMMAND ----------

housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# COMMAND ----------

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# COMMAND ----------

test_set.head()

# COMMAND ----------

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# COMMAND ----------

print(train_set.size, test_set.size, housing.size)
print(train_set["median_house_value"].mean())
print(test_set["median_house_value"].mean())

# COMMAND ----------

test_set.head()

# COMMAND ----------

housing["median_income"].hist()

# COMMAND ----------

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# COMMAND ----------

housing["income_cat"].value_counts()

# COMMAND ----------

housing["income_cat"].hist()

# COMMAND ----------

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# COMMAND ----------

strat_test_set["income_cat"].value_counts() / len(strat_test_set)

# COMMAND ----------

housing["income_cat"].value_counts() / len(housing)

# COMMAND ----------

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

# COMMAND ----------

compare_props

# COMMAND ----------

for sets in (strat_train_set, strat_test_set):
    sets.drop("income_cat", axis=1, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Discover and visualize the data to gain insights

# COMMAND ----------

housing = strat_train_set.copy()

# COMMAND ----------

housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")

# COMMAND ----------

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")

# COMMAND ----------

# MAGIC %md
# MAGIC The argument `sharex=False` fixes a display bug (the x-axis values and legend were not displayed). This is a temporary fix (see: https://github.com/pandas-dev/pandas/issues/10611 ). Thanks to Wilmer Arellano for pointing it out.

# COMMAND ----------

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")

# COMMAND ----------

# Download the California image
images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

# COMMAND ----------

import matplotlib.image as mpimg
california_img=mpimg.imread(os.path.join(images_path, filename))
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                  s=housing['population']/100, label="Population",
                  c="median_house_value", cmap=plt.get_cmap("jet"),
                  colorbar=False, alpha=0.4)
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
plt.show()

# COMMAND ----------

corr_matrix = housing.corr()

# COMMAND ----------

corr_matrix["median_house_value"].sort_values(ascending=False)

# COMMAND ----------

# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")

# COMMAND ----------

import seaborn as sns
sns.pairplot(housing[["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]])

# COMMAND ----------

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")

# COMMAND ----------

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

# COMMAND ----------

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# COMMAND ----------

housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()

# COMMAND ----------

housing.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare the data for Machine Learning algorithms

# COMMAND ----------

housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()

# COMMAND ----------

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows

# COMMAND ----------

sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1

# COMMAND ----------

sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2

# COMMAND ----------

median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3

# COMMAND ----------

sample_incomplete_rows

# COMMAND ----------

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# COMMAND ----------

# MAGIC %md
# MAGIC Remove the text attribute because median can only be calculated on numerical attributes:

# COMMAND ----------

housing_num = housing.drop("ocean_proximity", axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])

# COMMAND ----------

imputer.fit(housing_num)

# COMMAND ----------

imputer.statistics_

# COMMAND ----------

# MAGIC %md
# MAGIC Check that this is the same as manually computing the median of each attribute:

# COMMAND ----------

housing_num.median().values

# COMMAND ----------

# MAGIC %md
# MAGIC Transform the training set:

# COMMAND ----------

X = imputer.transform(housing_num)

# COMMAND ----------

type(X)

# COMMAND ----------

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)

# COMMAND ----------

housing_tr.loc[sample_incomplete_rows.index.values]

# COMMAND ----------

imputer.strategy

# COMMAND ----------

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# COMMAND ----------

housing_tr.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's preprocess the categorical input feature, `ocean_proximity`:

# COMMAND ----------

housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

# COMMAND ----------

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:15]

# COMMAND ----------

ordinal_encoder.categories_

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

# COMMAND ----------

# MAGIC %md
# MAGIC By default, the `OneHotEncoder` class returns a sparse array, but we can convert it to a dense array if needed by calling the `toarray()` method:

# COMMAND ----------

housing_cat_1hot.toarray()

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, you can set `sparse=False` when creating the `OneHotEncoder`:

# COMMAND ----------

cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

# COMMAND ----------

cat_encoder.categories_

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create a custom transformer to add extra attributes:

# COMMAND ----------

from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that I hard coded the indices (3, 4, 5, 6) for concision and clarity in the book, but it would be much cleaner to get them dynamically, like this:

# COMMAND ----------

col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] # get the column indices

# COMMAND ----------

# MAGIC %md
# MAGIC Also, `housing_extra_attribs` is a NumPy array, we've lost the column names (unfortunately, that's a problem with Scikit-Learn). To recover a `DataFrame`, you could run this:

# COMMAND ----------

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's build a pipeline for preprocessing the numerical attributes:

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# COMMAND ----------

housing_num_tr

# COMMAND ----------

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# COMMAND ----------

list(housing)

# COMMAND ----------

housing_prepared

# COMMAND ----------

housing_prepared.shape

# COMMAND ----------

# MAGIC %md
# MAGIC For reference, here is the old solution based on a `DataFrameSelector` transformer (to just select a subset of the Pandas `DataFrame` columns), and a `FeatureUnion`:

# COMMAND ----------

from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's join all these components into a big pipeline that will preprocess both the numerical and the categorical features:

# COMMAND ----------

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

old_num_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

old_cat_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

# COMMAND ----------

from sklearn.pipeline import FeatureUnion

old_full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ])

# COMMAND ----------

old_housing_prepared = old_full_pipeline.fit_transform(housing)
old_housing_prepared

# COMMAND ----------

# MAGIC %md
# MAGIC The result is the same as with the `ColumnTransformer`:

# COMMAND ----------

np.allclose(housing_prepared, old_housing_prepared)

# COMMAND ----------

# MAGIC %md
# MAGIC # Select and train a model 

# COMMAND ----------

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# COMMAND ----------

housing.iloc[0:2, 0:2]

# COMMAND ----------

# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))

# COMMAND ----------

# MAGIC %md
# MAGIC Compare against the actual values:

# COMMAND ----------

print("Labels:", list(some_labels))

# COMMAND ----------

some_data_prepared

# COMMAND ----------

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# COMMAND ----------

# MAGIC %md
# MAGIC **Note**: since Scikit-Learn 0.22, you can get the RMSE directly by calling the `mean_squared_error()` function with `squared=False`.

# COMMAND ----------

from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

# COMMAND ----------

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# COMMAND ----------

# MAGIC %md
# MAGIC # Fine-tune your model

# COMMAND ----------

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

# COMMAND ----------

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

# COMMAND ----------

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# COMMAND ----------

# MAGIC %md
# MAGIC **Note**: we specify `n_estimators=100` to be future-proof since the default value is going to change to 100 in Scikit-Learn 0.22 (for simplicity, this is not shown in the book).

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

# COMMAND ----------

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

# COMMAND ----------

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# COMMAND ----------

scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()

# COMMAND ----------

from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse

# COMMAND ----------

import joblib
joblib.dump(svm_reg, "test.pkl")

# COMMAND ----------



# COMMAND ----------

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [10, 30, 50], 'max_features': [4, 6, 8, 10]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC The best hyperparameter combination found:

# COMMAND ----------

grid_search.best_params_

# COMMAND ----------

grid_search.best_estimator_

# COMMAND ----------

# MAGIC %md
# MAGIC Let's look at the score of each hyperparameter combination tested during the grid search:

# COMMAND ----------

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# COMMAND ----------

pd.DataFrame(grid_search.cv_results_)

# COMMAND ----------

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

# COMMAND ----------

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# COMMAND ----------

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

# COMMAND ----------

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# COMMAND ----------

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# COMMAND ----------

final_rmse

# COMMAND ----------

# MAGIC %md
# MAGIC We can compute a 95% confidence interval for the test RMSE:

# COMMAND ----------

from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))

# COMMAND ----------

# MAGIC %md
# MAGIC We could compute the interval manually like this:

# COMMAND ----------

m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, we could use a z-scores rather than t-scores:

# COMMAND ----------

zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)

# COMMAND ----------

# MAGIC %md
# MAGIC # Extra material

# COMMAND ----------

# MAGIC %md
# MAGIC ## A full pipeline with both preparation and prediction

# COMMAND ----------

full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model persistence using joblib

# COMMAND ----------

my_model = full_pipeline_with_predictor

# COMMAND ----------

import joblib
joblib.dump(my_model, "my_model.pkl") # DIFF
#...
my_model_loaded = joblib.load("my_model.pkl") # DIFF

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example SciPy distributions for `RandomizedSearchCV`

# COMMAND ----------

from scipy.stats import geom, expon
geom_distrib=geom(0.5).rvs(10000, random_state=42)
expon_distrib=expon(scale=1).rvs(10000, random_state=42)
plt.hist(geom_distrib, bins=50)
plt.show()
plt.hist(expon_distrib, bins=50)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Exercise solutions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.

# COMMAND ----------

# MAGIC %md
# MAGIC Question: Try a Support Vector Machine regressor (`sklearn.svm.SVR`), with various hyperparameters such as `kernel="linear"` (with various values for the `C` hyperparameter) or `kernel="rbf"` (with various values for the `C` and `gamma` hyperparameters). Don't worry about what these hyperparameters mean for now. How does the best `SVR` predictor perform?

# COMMAND ----------

# MAGIC %md
# MAGIC **Warning**: the following cell may take close to 30 minutes to run, or more depending on your hardware.

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(housing_prepared, housing_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC The best model achieves the following score (evaluated using 5-fold cross validation):

# COMMAND ----------

negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse

# COMMAND ----------

# MAGIC %md
# MAGIC That's much worse than the `RandomForestRegressor`. Let's check the best hyperparameters found:

# COMMAND ----------

grid_search.best_params_

# COMMAND ----------

# MAGIC %md
# MAGIC The linear kernel seems better than the RBF kernel. Notice that the value of `C` is the maximum tested value. When this happens you definitely want to launch the grid search again with higher values for `C` (removing the smallest values), because it is likely that higher values of `C` will be better.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.

# COMMAND ----------

# MAGIC %md
# MAGIC Question: Try replacing `GridSearchCV` with `RandomizedSearchCV`.

# COMMAND ----------

# MAGIC %md
# MAGIC **Warning**: the following cell may take close to 45 minutes to run, or more depending on your hardware.

# COMMAND ----------

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC The best model achieves the following score (evaluated using 5-fold cross validation):

# COMMAND ----------

negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse

# COMMAND ----------

# MAGIC %md
# MAGIC Now this is much closer to the performance of the `RandomForestRegressor` (but not quite there yet). Let's check the best hyperparameters found:

# COMMAND ----------

rnd_search.best_params_

# COMMAND ----------

# MAGIC %md
# MAGIC This time the search found a good set of hyperparameters for the RBF kernel. Randomized search tends to find better hyperparameters than grid search in the same amount of time.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's look at the exponential distribution we used, with `scale=1.0`. Note that some samples are much larger or smaller than 1.0, but when you look at the log of the distribution, you can see that most values are actually concentrated roughly in the range of exp(-2) to exp(+2), which is about 0.1 to 7.4.

# COMMAND ----------

expon_distrib = expon(scale=1.)
samples = expon_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Exponential distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The distribution we used for `C` looks quite different: the scale of the samples is picked from a uniform distribution within a given range, which is why the right graph, which represents the log of the samples, looks roughly constant. This distribution is useful when you don't have a clue of what the target scale is:

# COMMAND ----------

reciprocal_distrib = reciprocal(20, 200000)
samples = reciprocal_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Reciprocal distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The reciprocal distribution is useful when you have no idea what the scale of the hyperparameter should be (indeed, as you can see on the figure on the right, all scales are equally likely, within the given range), whereas the exponential distribution is best when you know (more or less) what the scale of the hyperparameter should be.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.

# COMMAND ----------

# MAGIC %md
# MAGIC Question: Try adding a transformer in the preparation pipeline to select only the most important attributes.

# COMMAND ----------

from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

# COMMAND ----------

# MAGIC %md
# MAGIC Note: this feature selector assumes that you have already computed the feature importances somehow (for example using a `RandomForestRegressor`). You may be tempted to compute them directly in the `TopFeatureSelector`'s `fit()` method, however this would likely slow down grid/randomized search since the feature importances would have to be computed for every hyperparameter combination (unless you implement some sort of cache).

# COMMAND ----------

# MAGIC %md
# MAGIC Let's define the number of top features we want to keep:

# COMMAND ----------

k = 5

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's look for the indices of the top k features:

# COMMAND ----------

top_k_feature_indices = indices_of_top_k(feature_importances, k)
top_k_feature_indices

# COMMAND ----------

np.array(attributes)[top_k_feature_indices]

# COMMAND ----------

# MAGIC %md
# MAGIC Let's double check that these are indeed the top k features:

# COMMAND ----------

sorted(zip(feature_importances, attributes), reverse=True)[:k]

# COMMAND ----------

# MAGIC %md
# MAGIC Looking good... Now let's create a new pipeline that runs the previously defined preparation pipeline, and adds top k feature selection:

# COMMAND ----------

preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])

# COMMAND ----------

housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's look at the features of the first 3 instances:

# COMMAND ----------

housing_prepared_top_k_features[0:3]

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's double check that these are indeed the top k features:

# COMMAND ----------

housing_prepared[0:3, top_k_feature_indices]

# COMMAND ----------

# MAGIC %md
# MAGIC Works great!  :)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.

# COMMAND ----------

# MAGIC %md
# MAGIC Question: Try creating a single pipeline that does the full data preparation plus the final prediction.

# COMMAND ----------

prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('svm_reg', SVR(**rnd_search.best_params_))
])

# COMMAND ----------

prepare_select_and_predict_pipeline.fit(housing, housing_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try the full pipeline on a few instances:

# COMMAND ----------

some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))

# COMMAND ----------

# MAGIC %md
# MAGIC Well, the full pipeline seems to work fine. Of course, the predictions are not fantastic: they would be better if we used the best `RandomForestRegressor` that we found earlier, rather than the best `SVR`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.

# COMMAND ----------

# MAGIC %md
# MAGIC Question: Automatically explore some preparation options using `GridSearchCV`.

# COMMAND ----------

# MAGIC %md
# MAGIC **Warning**: the following cell may take close to 45 minutes to run, or more depending on your hardware.

# COMMAND ----------

param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
                                scoring='neg_mean_squared_error', verbose=2)
grid_search_prep.fit(housing, housing_labels)

# COMMAND ----------

grid_search_prep.best_params_

# COMMAND ----------

# MAGIC %md
# MAGIC The best imputer strategy is `most_frequent` and apparently almost all features are useful (15 out of 16). The last one (`ISLAND`) seems to just add some noise.

# COMMAND ----------

# MAGIC %md
# MAGIC Congratulations! You already know quite a lot about Machine Learning. :)
