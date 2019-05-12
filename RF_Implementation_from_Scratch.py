#Import Dependencies
from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics

PATH = "data/bulldozers"

df_raw = pd.read_feather("tmp/bulldozers-raw")
df_trn, y_trn, nas = proc_df(df_raw, "SalePrice")

def split_vals(a, n):
  return a[:n], a[n:]

n_valid = 12000
n_trn = len(df_trn) - n_valid
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)

x_sub = X_train[['YearMade', 'MachineHoursCurrentMeter']]


class TreeEnsemble():
  def __init__(self, x, y, n_trees, sample_sz, min_leaf = 5):
    np.random.seed(1)
    self.x, self.y, self.sample_sz, self.min_leaf = x, y, sample_sz, min_leaf
    self.trees = [self.create_tree() for i in range(n_trees)]
  
  def create_tree():
    rnd_idxs = np.random.permutation(len(self.y))[:self.sample_sz]
    return DecisionTree(self.x.iloc[rnd_idxs], self.y.iloc[rnd_idxs], min_leaf = self.min_leaf)
  
  def predict():
    return np.mean([t.predict(x) for t in self.trees], axis = 0)

class DecisionTree():
  def __init__(self, x, y, idxs = None, min_leaf = 5):
    if idxs is None:
      idxs = np.arange(len(y))
    self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf
    self.n, self.c = len(idxs), x.shape[1]
    self.val = np.mean(y[idxs])
    self.score = float('inf')
    self.find_varsplit()
  
  def find_varsplit(self):
    for i in range(self.c):
      self.find_better_split(i)
  
  def find_better_split(self, var_idx):
    pass
  
  @property
  def split_name():
  
  @property
  def split_col():
    return self.x.values[self.idxs, self.var_idx]
  
  @property
  def is_leaf(self):
    return self.score == float("inf")
  
  def __repr__():
    s = f'n: {self.n}; val: {self.val}'
    if not self.is_leaf:
      s += f'; score: {self.score}; split {self.split}; var: {self.split_name}'
    return s

m = TreeEnsemble(X_train, y_train, n_trees = 10, sample_sz = 1000, min_leaf = 3)


ens = TreeEnsemble(x_sub, y_train, 1, 1000)
tree = ens.trees[0]
x_samp, y_samp = tree.x, tree.y

m = RandomForestRegressor(n_estimators = 1, max_depth = 1, bootstrap = False)
m.fit(x_samp, y_samp)
draw_tree(m.estimators_[0], x_samp, precision = 2)

def find_better_split(self, var_idx):
  x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
  
  for i in range(1, self.n-1):
    lhs = (x <= x[i])
    rhs = (x > x[i])
    if rhs.sum() == 0:
      continue
    lhs_std = y[lhs].std()
    rhs_std = y[rhs].std()
    curr_score = lhs_std*lhs.sum() + rhs_std*rhs.sum()
    
    if curr_score < self.score:
      self.var_idx, self.score, self.split = var_idx, curr_score, x[i]

%timeit find_better_split(tree, 1)
tree

tree = TreeEnsemble(x_sub, y_train, 1, 1000).trees[0]

def std_agg(cnt, s1, s2):
  return math.sqrt((s2 / cnt) - (s1 / cnt)**2)

def find_better_split_foo(self, var_idx):
  x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
  
  sort_idx = np.argsort(x)
  sort_x, sort_y = x[sort_idx], y[sort_idx]
  rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
  lhs_cnt, lhs_sum, lhs_sum2 = 0, 0., 0.
  
  for i in range(0, self.n - self.min_leaf - 1):
    xi, yi = sort_x[i], sort_y[i]
    lhs_cnt += 1; rhs_cnt -= 1
    lhs_sum += yi; rhs_sum -= yi
    lhs_sum2 += yi**2; rhs_sum2 -= yi**2
    if i < self.min_leaf or xi == sort_x[i+1]:
      continue
    
    lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
    rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
    curr_score = lhs_std * lhs_cnt + rhs_std * rhs_cnt
    if curr_score < self.score:
      self.var_idx, self.score, self.split = var_idx, curr_score, xi

%timeit find_better_split_foo(tree, 1)
tree

DecisionTree.find_better_split = find_better_split_foo


def find_varsplit(self):
  for i in range(self.c):
    self.find_better_split(i)
  if self.is_leaf:
    return
  x = self.split_col
  lhs = np.nonzero(x <= self.split)[0]
  rhs = np.nonzero(x > self.split)[0]
  self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs])
  self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs])

DecisionTree.find_varsplit = find_varsplit

def predict(self, x):
  return np.array([self.predict_row(xi) for xi in x])

def predict_row():
  if self.is_leaf:
    return self.val
  t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
  return t.predict_row(xi)

DecisionTree.predict_row = predict_row

DecisionTree.predict = predict

%time preds = tree.predict(X_valid[cols].values)

plt.scattar(preds, y_valid, alpha = 0.05)

metrics.r2_score(preds, y_valid)

#Compare this with the Scikit Learn Random Forest.
m = RandomForestRegressor(n_estimators = 1, min_samples_leaf = 5, bootstrap = False)
%time m.fit(x_samp, y_samp)
preds = m.predict(X_valid[cols].values)
plt.scattar(preds, y_valid, alpha = 0.05)
metrics.r2_score(preds, y_valid)
