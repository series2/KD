import pandas as pd
import os
import re
import statistics
import yaml
import pickle

task_to_metrics = {
    "cola": 'matthews_correlation',
    "mnli": 'accuracy',
    "mrpc": 'f1',
    "qnli": 'accuracy',
    "qqp": 'accuracy',
    "rte": 'accuracy',
    "sst2": 'accuracy',
    "stsb": 'pearson',
    "wnli": 'accuracy',
}

config_filename = input('input your config filename or path. (ex sample.yaml : ')
with open(config_filename, 'r') as f:
    conf = yaml.safe_load(f)
# tasks
# directories
# max_checkpoint
#   true : max checkpoint
#   false : last
maindir = os.getcwd()
results = pd.DataFrame(index=conf['directories'], columns=conf['tasks'])
results_d = pd.DataFrame(index=conf['directories'], columns=conf['tasks'])
method_task_values = {}
for directory in conf['directories']:
    for task in conf['tasks']:
        path = os.path.join(maindir, directory, 'df_logs', task)
        # os.chdir(maindir)
        os.chdir(path)


        by_lr_values = {}
        # by_lr : {lr0: average of max epoch in lr0, lr1:...}
        by_lr = {}
        by_lr_d = {}
        for file in os.listdir():
            if file.endswith('.csv'):
                df = pd.read_csv(file)
                # calculate max or last metrics
                if conf['max_checkpoint']:
                    res_value = df[task_to_metrics[task]].max()
                else:
                    res_value = df.iloc[-1][task_to_metrics[task]]

                lr = float(re.sub('_\d+.csv', '', file))
                if lr not in by_lr_values:
                    by_lr_values[lr] = []
                if lr not in by_lr_d:
                    by_lr_d[lr] = 0
                if lr not in by_lr:
                    by_lr[lr] = 0
                by_lr_values[lr].append(res_value)
        for lr in by_lr:
            by_lr_d[lr] = statistics.stdev(by_lr_values[lr])
            by_lr[lr] = sum(by_lr_values[lr])/len(by_lr_values[lr])
        
        max_value = max(by_lr.values())
        max_lr = max(by_lr, key=by_lr.get)

        results.at[directory, task] = max_value
        results_d.at[directory, task] = by_lr_d[max_lr]

        if task not in method_task_values:
            method_task_values[task] = {}
        method_task_values[task][directory] = by_lr_values[max_lr]

print(results)
os.chdir(maindir)
results.to_csv(conf['outfile']+'.csv')
results_d.to_csv(conf['outfile']+'_d.csv')
#print(method_task_values)
with open(conf['outfile']+'.pickle', 'wb') as f:
    pickle.dump(method_task_values, f)





            
            


