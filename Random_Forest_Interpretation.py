#### Confidence based on Tree Variance
set_rf_samples(50000)

m = RandomForestRegressor(n_estimators = 40, min_samples_leaf = 3, max_features = 0.5, n_jobs = -1, oob_score = True)
m.fit(X_trian, y_train)
print_score(m)


#### Feature Importance (Two ways - one is the below one and the other is with the help of Tree Interpreter)
to_keep = fi[fi.imp >0.005].cols;
len(to_keep)


#### Remove Redundant Features
from scipy.cluster import hierarchy as hierarchy

corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1 - corr)
z = hc.linkage(corr_condensed, method = "average")
fig = plt.figure(figsize = (16, 10))
dendrogram = hc.dendrogram(z, labels = df_keep.columns, orientation = "left", leaf_font_size = 16)
plt.show()

#Out Of Bag Score (OOB Score)
def get_oob(df):
  m = RandomForestRegressor(n_estimators = 40, min_samples_leaf = 3, max_features = 0.5, n_jobs = -1, oob_score = True)
  x, _ split_vals(df, n_trn)
  m.fit(x, y_train)
  return m.oob_score
get_oob(df_keep)

to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_oob(df_keep.drop(to_drop, axis = 1))


#### Parial Dependence
from pdpbox import pdp
from plotnine import *

x_all = get_sample(df_raw[df_raw.YearMade > 1930], 500)

gg_plot(x_all, aes("YearMade", "SalePrice")) + stat_smooth(se = True, method = "loess")

x = get_sample(X_train[X_train.YearMade > 1930], 500)

def plot_pdp(feat, clusters = None, feat_name = None):
  feat_name feat_name or feat
  p = pdp.pdp_isolate(m, x, feat)
  
  return pdp.pdp_plot(p, feat_name, plot_lines = True, cluster = clusters is not None, n_cluster_centers = clusters)

plot_pdp("YearMade")
plot_pdp("YearMade", clusters = 5)

#We can try plotting feature importance on the basis of taking the categories and creating a one hot vector of each category as a column
#The other way is to take two features and make them as one by taking there difference for example
#By doing this you might get new features which might turn out to be most important


#### Tree Interpreter
from treeinterpreter import treeinterpreter as ti
df_train, df_valid = split_vals(df_raw[df_keep.columns], n_trn)

row = X_valid.values[None, 0]
row

prediction, bias, contributions = ti.predict(m, row)

prediction[0], bias[0]
idxs = np.argsort(contributions[0])
[o for o in zip(df_keep.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]
