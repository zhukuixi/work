# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:31:41 2023

@author: Kuixi Zhu
"""

import pandas as pd

gene_53 = pd.read_csv("D:/Dropbox/TIME/2023/1106_CoxModel/53GenePanel/53gene.csv")
gene_53.FinalGeneName.replace({'HLA.DPA1':'HLA-DPA1','HLA.DPB1':'HLA-DPB1','HLA.DQB1':'HLA-DQB1'},inplace=True)
feature = list(gene_53.iloc[:,0]) + ['Survival','event_occurred']
feature_gene = list(gene_53.iloc[:,0])

# sinai
data_sinai = pd.read_csv("D:/Dropbox/TIME/2023/1106_CoxModel/NormalizedData/SinaiData/TrainingDataFinal.csv")
data_sinai.set_index('SampleIndex',inplace=True)
survival_sinai = pd.read_csv("D:/Dropbox/TIME/2023/1106_CoxModel/NormalizedData/SinaiData/SINAIDISCOVERYFORRUI2023_Clinic.csv",skiprows=[1,2],nrows=44)
survival_sinai = survival_sinai.loc[:,['RCC Name','Survival','Died Melanoma']]
survival_sinai['Died Melanoma'].replace({'unk':0},inplace=True)
survival_sinai.rename({'Died Melanoma':'event_occurred'},axis=1,inplace=True)
data_sinai = data_sinai.reset_index()
data_sinai = data_sinai.merge(survival_sinai,left_on='SampleIndex',right_on = 'RCC Name')
data_sinai.set_index('SampleIndex',inplace=True)

# columbia
data_columbia  = pd.read_csv("D:/Dropbox/TIME/2023/1106_CoxModel/NormalizedData/ColumbiaData/Columbia.Normalized_ys_4HK.csv")
data_columbia = data_columbia.set_index('Unnamed: 0').T
data_columbia = data_columbia.reset_index()
data_columbia.rename({'index':'SampleIndex','CD180':'LY64','HLA-E':'HLAE','IL37':'IL1F7'},axis=1,inplace=True)
data_columbia.set_index('SampleIndex',inplace=True)

survival_columbia = pd.read_csv('D:/Dropbox/TIME/2023/1106_CoxModel/NormalizedData/ColumbiaData/Merged_Normalized_covariate_Columbia.csv')
survival_columbia = survival_columbia.loc[:,['SampleID','Survival Months','Died of Melanoma (follow up>36 months)']]
survival_columbia['Died of Melanoma (follow up>36 months)'].replace({'unk':0},inplace=True)
survival_columbia.rename({'Survival Months':'Survival','Died of Melanoma (follow up>36 months)':'event_occurred'},axis=1,inplace=True)
data_columbia = data_columbia.reset_index()
data_columbia = data_columbia.merge(survival_columbia,left_on='SampleIndex',right_on = 'SampleID')
data_columbia.set_index('SampleIndex',inplace=True)

# roswell
data_rosewell = pd.read_csv("D:/Dropbox/TIME/2023/1106_CoxModel/NormalizedData/RoswellData/Roswell.Normalized_ys_4HK.csv")
data_rosewell = data_rosewell.set_index('Unnamed: 0').T
data_rosewell = data_rosewell.reset_index()
data_rosewell.rename({'index':'SampleIndex','CD180':'LY64','HLA-E':'HLAE','IL37':'IL1F7'},axis=1,inplace=True)
data_rosewell.set_index('SampleIndex',inplace=True)
data_rosewell = data_rosewell.loc[:,list(gene_53.iloc[:,0])]


#set(gene_53.iloc[:,0]) - set(data_columbia.columns.values)
#set(gene_53.iloc[:,0]) - set(data_sinai.columns.values)
#set(gene_53.iloc[:,0]) - set(data_rosewell.columns.values)

# outlier

outlier = pd.read_csv('D:/Dropbox/TIME/2023/1106_CoxModel/OutlierIDs.csv')
outlier_sample = outlier.iloc[:,1].to_list()

data_sinai= data_sinai.loc[:,feature]
data_columbia = data_columbia.loc[:,feature]
data_combo = pd.concat([data_sinai,data_columbia])
data_combo['event_occurred'] = data_combo.event_occurred.astype('int')


removed_sample = ["X.X20230503_1.1.31021038722_T20210528.1_01.RCC.",

  "X.X20230615_2.31020999446_s.mip53_01.RCC.",

  "X.X20230615_3.31020999447_s.mip53_01.RCC.",

  "X.X20230616_4.31020999448_s.mip53L2_01.RCC.",

  "X.X20230623_5.31020999449_MIP52L2_01.RCC.",

  "X.X20230623_6.31020999471_MIP52L2_01.RCC.",

  "X.X20230711_7.31021038724_T20210528.1_01.RCC.",

  "X.X20230711_8.31021038725_T20210528.1_01.RCC.",

  "X.X20230712_10.31021038677_T20210528.1_01.RCC.",

  "X.X20230712_9.31021038676_T20210528.1_01.RCC.",

  "X.X20230713_11_T20210528.1_01.RCC.",

  "X.X20230713_12.31021038629_T20210528.1_01.RCC.",

  "X.X20230714_13.31021038627_T20210528.1_01.RCC.",

  "X.X20230714_14.31021038700_T20210528.1_01.RCC.",
   "X.X20230615_3.31020999447_9_05.RCC.",

  "X.X20230616_4.31020999448_41b_10.RCC.",

  "X.X20230711_8.31021038725_18C_08.RCC.",

  "X.X20230714_14.31021038700_49d_10.RCC."]



removed_sample = [sample[2:-5].replace('.','-')+".RCC" for sample in removed_sample]
# remove samples
data_rosewell = data_rosewell.loc[~data_rosewell.index.isin(removed_sample),]

########


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
from lifelines.utils.sklearn_adapter import sklearn_adapter

from sklearn.model_selection import GridSearchCV



cph = CoxPHFitter()
cph.fit(data_combo, duration_col='Survival', event_col='event_occurred',show_progress=True)


cph = CoxPHFitter(penalizer=0.01, l1_ratio=0)
cph.fit(data_combo.loc[~data_combo.index.isin(outlier_sample),:], duration_col='Survival', event_col='event_occurred')


from lifelines.datasets import load_rossi

rossi = load_rossi()

