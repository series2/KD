from scipy import stats
import pandas as pd
import os
import re
import statistics
import pickle
import itertools
import yaml

config_filename = input('input your config filename or path. (ex sample.yaml : ')
with open(config_filename, 'r') as f:
    conf = yaml.safe_load(f)

with open(conf['mtv'], 'rb') as f:
    method_taks_values = pickle.load(f)

significant = {}

def statistic(x, y):
    return sum(x)/len(x) - sum(y)/len(y)

p = 0.05
result = pd.read_csv(conf['result'], index_col=0)
print(result)
methods = None
all_c = None
for task in method_taks_values.keys():
    if methods is None:
        methods = list(method_taks_values[task].keys())
        for method in methods:
            if 'teacher' in method:
                methods.remove(method)
    if task not in significant:
        significant[task] = {}
        for method in methods:
            significant[task][method] = []
    if all_c is None:
        # combination of methods
        all_c = list(itertools.combinations(methods, 2))
    
    for c in all_c:
        # compare average
        if result.loc[c[0], task] >= result.loc[c[1], task]:
            c_rev = (c[0], c[1])
        else:
            c_rev = (c[1], c[0])
        large = method_taks_values[task][c_rev[0]]
        small = method_taks_values[task][c_rev[1]]
            
        #ttest_res = stats.ttest_ind(large, small, equal_var=False, alternative='greater')
        ptest_res = stats.permutation_test((large, small), statistic, alternative='greater')
        if ptest_res.pvalue < p:
            significant[task][c_rev[0]].append(c_rev[1])

for task in significant.keys():
    print(task)
    for method in methods:
        print(' ', method, ':', significant[task][method])

    