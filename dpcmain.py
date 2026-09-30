# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:09:58 2024

@author: User
"""
import os
import pandas as pd


dirpath = os.getcwd()

folder_path = dirpath + '\\data\\'

# file = ['2d-4c-no9.arff','jain.arff',  'donut3.arff','donutcurves.arff','2d-4c-no4.arff','Iris']

file = ['Iris']


temp = ord('a')

save_path = dirpath + "\\result_ph\\" + "accuracy_results.csv"  # 檔案儲存路徑


df = pd.DataFrame(columns=["id", "file_name", "Total Points", "Wrongly Merged Points", "Accuracy","Wrongly Split Points","Split Accuracy"])



print(f"Empty CSV created at {save_path}")

""" 指定群數 t """
t = 3 


import mydpc as dcl
for file_name in file:
    filein = folder_path + file_name + '.csv'
    dcl.run(fi=filein, sep=',',name = chr(temp))
    temp += 1
