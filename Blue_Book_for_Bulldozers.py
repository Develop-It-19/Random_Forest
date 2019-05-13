%load_ext autoreload
%autoreload 2
%matplotlib inline

!pip install fastai == 0.7.0

#Import Dependencies
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics

PATH = "/content/gdrive/My Drive/Colab Notebooks/data/bulldozers/"

!ls {PATH}

df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory = False, parse_dates = ["saledate"])

def display_all(df):
  with pd.option_context("display.max_rows", 1000):
    with pd.option_context("display.max_columns", 1000):
      display(df)
display_all(df_raw.tail().transpose())

df_raw.SalePrice = np.log(df_raw.SalePrice)

m = RandomForestRegressor(n_job = -1)

m.fit(df_raw.drop("SalePrice", axis = 1), df_raw.SalePrice)

add_datepart(df_raw, "saledate")
df_raw.saleYear.head()
df_raw.saleDayofweek.head()

train_cats(df_raw)

df_raw.UsageBand.cat.categories
df_raw.UsageBand.cat.set_categories(["High", "Medium", "Low"], ordered = True, inplace = True)

display_all(df_raw.tail())

df, y, nas = proc_df(df_raw, "SalePrice")

df.columns

#Train a RF on (df, y)
m = RandomForestRegressor(n_jobs = -1)
m.fit(df, y)
m.score(df, y)

def split_vals(a, n):
  return a[:n].copy(), a[n:].copy()

n_valid = 12000
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape

def rmse(x, y):
  return math.sqrt(((x - y)**2).mean())
  
def print_score(m):
  res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid), m.score(X_train, y_train), m.score(X_valid, y_valid)]
  if hasattr(m, "oob_score_"):
    res.append(m.oob_score_)
  print(res)

#Train a RF on (X_train, y_train) with training set size len(df) - 12000.
m = RandomForestRegressor(n_jobs = -1)
%time m.fit(X_train, y_train)
print_score(m)


#Using Subset of Data.
df_trn, y_trn, _ = proc_df(df_raw, "SalePrice", subset = 30000)
X_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)

#Train a Single Tree without Bootstrap
m = RandomForestRegressor(n_jobs = -1, n_estimators = 1, max_depth = 3, bootstrap = False)
m.fit(X_train, y_train)
print_score(m)

draw_tree(m.estimators_[0], X_train, precision = 3)

#Train 10 Trees.
m = RandomForestRegressor(n_jobs = -1)
%time m.fit(X_train, y_train)
print_score(m)

preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:, 0], np.mean(preds[:, 0]), y_valid[0]
preds.shape

plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis = 0)) for i in range(10)])

#Training 20 Trees with OOB.
m = RandomForestRegressor(n_jobs = -1, n_estimators = 20, oob_score = True)
m.fit(X_train, y_train)
print_score(m)


#Handling Over-Fitting
	#Sub-sampling
	#We take (X_train, y_train) from entire data (df, y) and take samples from it by Bootstapping for each tree.

df, y, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

set_rf_samples(20000)

num_trees = [40, 100, 200, 400, 4000]

for k in num_trees:

	m = RandomForestRegressor(n_jobs = -1, n_estimators = k, oob_score = False)
	%time m.fit(X_train, y_train)
	print_score(m)

	preds = np.stack([t.predict(X_valid) for t in m.esimators_])
	preds[:, 0], np.mean(preds[:, 0]), y_valid[0]
	preds.shape

	plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis = 0)) for i in range(k)])


#Running with 40000 decision trees with 40000 samples
set_rf_samples(40000)

m = RandomForestRegressor(n_jobs = -1, n_estimators = 40000, oob_score = False)
%time m.fit(X_train, y_train)
print_score(m)

preds = np.stack([t.predict(X_valid) for t in m.esimators_])
preds[:, 0], np.mean(preds[:, 0]), y_valid[0]
preds.shape

plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis = 0)) for i in range(40000)])


#Train using Extra Trees Regressor
from sklearn.ensemble import ExtraTreesRegressor
e = ExtraTreesRegressor(n_jobs = -1)
e.fit(X_train, y_train)
print_score(e)


#Tree building parameters
m = RandomForestRegressor(n_jobs = -1, min_samples_leaf = 3, n_estimators = 40, oob_score = True)
%time m.fit(X_train, y_train)
print_score(m)

m = RandomForestRegressor(n_jobs = -1, min_samples_leaf = 3, max_features = 0.7, n_estimators = 40, oob_score = True)
%time m.fit(X_train, y_train)
print_score(m)

