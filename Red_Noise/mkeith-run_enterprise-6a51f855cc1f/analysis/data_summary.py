#!/usr/bin/env python
'''Read pulsar glitches solution files and plot the correlation between parameters.
    Written by Y.Liu (yang.liu-50@postgrad.manchester.ac.uk).'''

import glob
import pandas as pd

file_name_list = glob.glob('slt_*.csv')
file_name_list.sort()
pulsar_name = [file_name.split('_')[-1].split('.')[0] for file_name in file_name_list]

summary = 'summary.csv'
sum_table = pd.DataFrame()
for i, solution in enumerate(file_name_list):
    table = pd.read_csv(solution, index_col=[0,1])
    sum_table = pd.concat([sum_table, table])
sum_table.to_csv(summary)
