from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler, RobustScaler, MaxAbsScaler
import torch
import numpy as np

from agentMET4FOF_ml_extension.agentMET4FOF.agentMET4FOF.streams import DataStreamMET4FOF
from baetorch.baetorch.models.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models.bae_hydra import BAE_Hydra, Hydra_Autoencoder
from baetorch.baetorch.models.base_autoencoder import Encoder, infer_decoder, Autoencoder
from baetorch.baetorch.models.base_layer import DenseLayers
from baetorch.baetorch.models.cholesky_layer import CholLayer
from baetorch.baetorch.plotting import get_grid2d_latent, plot_contour
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.misc import save_bae_model, load_bae_model
from baetorch.baetorch.util.seed import bae_set_seed
from sklearn import metrics

from util.calc_auroc import calc_auroc_score
from util.calc_outlier import get_outlier_arg, multi_intersection_combo, get_num_outliers_df
from util.calc_picp import calc_picp, calc_perc_boundary
from util.plotting import plot_tornado, plot_machines_tornado

bae_set_seed(1231)

# dataset_folder = "multi-stage-dataset/"
# df = pd.read_csv(dataset_folder+"continuous_factory_process.csv")
#
#
# model_stage = 2
# train_model = True
#
# #drop columns
# df = df.drop(["time_stamp"], axis=1)
# column_names = df.columns
#
# if model_stage ==1:
#     Y_df = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Actual" in name)]]
# else:
#     Y_df = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Actual" in name)]]
# # Y = Y[[name for name in Y if "Actual" in name]]
#
# if model_stage ==1:
#     X_df = df[[col_name for col_name in df.columns if ("Machine1" in col_name) or ("Machine2" in col_name) or ("Machine3" in col_name)]]
# else:
#     X_df = df[[col_name for col_name in df.columns if ("Machine4" in col_name) or ("Machine5" in col_name)]]
#
#
# X_columns = X_df.columns
# X = X_df.values[:,:]
# Y = Y_df.values
#
# upper_quartile = 95
# lower_quartile = 100-upper_quartile
# total_y_dims = Y.shape[-1]
# unhealthy_indices = [np.argwhere((Y[:,i]>=np.percentile(Y[:,i],upper_quartile)) | (Y[:,i]<=np.percentile(Y[:,i],lower_quartile))).squeeze(-1) for i in np.arange(total_y_dims)]
#
# unhealthy_index = unhealthy_indices[0]
# for unhealthy_index_temp in unhealthy_indices:
#     unhealthy_index = np.intersect1d(unhealthy_index,unhealthy_index_temp)
#
# healthy_index = np.array([i for i in np.arange(Y.shape[0]) if i not in unhealthy_index])
#
#
# #========================FILTER HEALTHY AND UNHEALTHY DATA SPLIT===========
#
# x_train = X[healthy_index]
# y_train = Y[healthy_index]
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.5)
# x_ood = X[unhealthy_index]
# y_ood =  Y[unhealthy_index]
#
#
# homoscedestic_mode = "every"
#
# #==========================APPLY SCALING AND CLIPPING========================
# scaler = MinMaxScaler()
#
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# x_ood = scaler.transform(x_ood)
#
# x_test = np.clip(x_test,0,1)
# x_ood = np.clip(x_ood,0,1)
#
# print("STAGE :"+str(model_stage))
# print("X TRAIN SHAPE: "+ str(x_train.shape))
# print("X TEST SHAPE: "+ str(x_test.shape))
# print("X OOD SHAPE: "+ str(x_ood.shape))
#
# print("Y TRAIN SHAPE: "+ str(y_train.shape))
# print("Y TEST SHAPE: "+ str(y_test.shape))
# print("Y OOD SHAPE: "+ str(y_ood.shape))

#=============================================================================


class Liveline_DataStream(DataStreamMET4FOF):
    def __init__(self, dataset_folder = "multi-stage-dataset/",
                 output_stage=1,
                 input_stage=1,
                 upper_quartile=80, train_size=0.5,
                 apply_scaling=True):

        lower_quartile = 100 - upper_quartile
        df = pd.read_csv(dataset_folder+"continuous_factory_process.csv")

        #drop columns
        df = df.drop(["time_stamp"], axis=1)
        column_names = df.columns

        # filter Y columns
        if output_stage == 1:
            Y_df_actual = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Actual" in name)]]
            Y_df_setpoint = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Setpoint" in name)]]
            Y_df_actual.columns =[str(i) for i in range(len(Y_df_actual.columns))]
            Y_df_setpoint.columns = [str(i) for i in range(len(Y_df_setpoint.columns))]
            Y_df = (Y_df_setpoint-Y_df_actual).abs()

            # Y_df = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Actual" in name)]]

        else:
            Y_df_actual = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Actual" in name)]]
            Y_df_setpoint = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Setpoint" in name)]]
            Y_df_actual.columns =[str(i) for i in range(len(Y_df_actual.columns))]
            Y_df_setpoint.columns = [str(i) for i in range(len(Y_df_setpoint.columns))]
            Y_df = (Y_df_setpoint-Y_df_actual).abs()

            # Y_df = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Actual" in name)]]
        # self.Y_df = Y_df
        # filter X columns
        if input_stage ==1:
            X_df = df[[col_name for col_name in df.columns if ("Machine1" in col_name) or ("Machine2" in col_name) or ("Machine3" in col_name)]]
        elif input_stage ==2:
            X_df = df[[col_name for col_name in df.columns if ("Machine4" in col_name) or ("Machine5" in col_name)]]
        else:
            X_df = df[[col_name for col_name in df.columns if ("Machine1" in col_name) or ("Machine2" in col_name) or ("Machine3" in col_name) or ("Machine4" in col_name) or ("Machine5" in col_name)]]

        X_columns = X_df.columns
        X = X_df.values
        Y = Y_df.values
        self.Y_df_vals = Y.copy()

        num_examples = Y.shape[0]
        self.num_examples = num_examples

        total_y_dims = Y.shape[-1]

        self.y_levels = get_num_outliers_df(self.Y_df_vals)


        unhealthy_index = (np.argwhere(self.y_levels>=1)).flatten()
        healthy_index = (np.argwhere(self.y_levels==0)).flatten()
        Y = self.y_levels

        x_train = X[healthy_index]
        y_train = Y[healthy_index]
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=train_size)
        x_ood = X[unhealthy_index]
        y_ood = Y[unhealthy_index]


        # apply Scaling
        if apply_scaling:
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            x_ood = scaler.transform(x_ood)

            x_test = np.clip(x_test, 0, 1)
            x_ood = np.clip(x_ood, 0, 1)

        print("INPUT STAGE :" + str(input_stage))
        print("OUTPUT STAGE :" + str(output_stage))

        print("X TRAIN SHAPE: " + str(x_train.shape))
        print("X TEST SHAPE: " + str(x_test.shape))
        print("X OOD SHAPE: " + str(x_ood.shape))

        print("Y TRAIN SHAPE: " + str(y_train.shape))
        print("Y TEST SHAPE: " + str(y_test.shape))
        print("Y OOD SHAPE: " + str(y_ood.shape))

        self.set_data_source(quantities={"train":x_train,"test":x_test,"ood":x_ood},
                             target={"train":y_train,"test":y_test,"ood":y_ood})

        # self.unhealthy_indices = unhealthy_indices
        self.X_columns = X_columns
        # self.y_outliers = y_outliers


liveline_datastream = Liveline_DataStream(output_stage=2)

# y_outliers : list of size = num_dimensions
# y_outliers_levelup = [(multi_intersection_combo(liveline_datastream.y_outliers, level = level+1)) for level in range(len(liveline_datastream.y_outliers))]
#
# joe = (multi_intersection_combo(liveline_datastream.y_outliers, level = 1))
# num_examples = liveline_datastream.num_examples
# y_levels = np.arange(num_examples) * 0
#
# for level in range(len(liveline_datastream.y_outliers)):
#     y_levels[y_outliers_levelup[level]] = level
#
# y_levels = y_levels
#
# np.sum([len(y_outlier) for y_outlier in y_outliers_levelup[1:]])


#
# def calc_outlier_range_(y_col, q1=0.25, q3=0.75):
#     Q1 =  np.quantile(y_col, q1, axis=0)
#     Q3 = np.quantile(y_col, q3, axis=0)
#     IQR = Q3 - Q1
#
#     return Q1,Q3,IQR
#
# def calc_outlier_ranges(y_df):
#     """
#     Returns
#     -------
#     np.array of (Q1, Q3, IQR) for each column of y_df.
#     Use this to determine outlier
#     """
#
#     return np.array([calc_outlier_range_(y_col) for y_col in y_df.T])
#
# def get_num_outliers(y_row, outlier_ranges, return_sum = True):
#     """
#     Parameters
#     ----------
#     y_row : A row of N columns, which obtained from y_df
#
#     outlier_ranges : np.array of (Q1, Q3, IQR) for each column of y_df. Use the output from `calc_outlier_ranges` method as this parameter.
#
#     Returns
#     -------
#     num_outlier_cols : (int) total number of columns with outliers
#
#     """
#     num_outlier_cols = (y_row < outlier_ranges[:, 0] - 1.5 * outlier_ranges[:, 2]) | (
#                 y_row > outlier_ranges[:, 1] + 1.5 * outlier_ranges[:, 2]).astype(int)
#     if return_sum:
#         num_outlier_cols = np.sum(num_outlier_cols)
#     return num_outlier_cols
#
# def get_num_outliers_df(y_df):
#     """
#     1. Determines the outlier ranges
#     2. For every row, determine number of columns in the y_df which is an outlier
#
#     Note: 0 means no outlier. The higher the number, the higher the `degree of outlierness` i.e defect.
#     """
#     outlier_ranges = calc_outlier_ranges(y_df)
#     num_outliers = np.apply_along_axis(get_num_outliers, axis=1, arr=y_df, outlier_ranges=outlier_ranges)
#     return num_outliers

y_df = liveline_datastream.Y_df_vals

num_outliers = get_num_outliers_df(y_df)
count_outliers = np.unique(num_outliers,return_counts=True)


# print(len(np.argwhere(num_outliers==14)))
pprint(count_outliers)
# def is_outlier(y_row,Q1,Q3):
#     IQR = Q3 - Q1
#     outlier_args = np.argwhere((y_col < Q1 - 1.5 * IQR) | (y_col > Q3 + 1.5 * IQR))


# def fit_outlier(y_df, q1=0.25, q3=0.75):





# y_outliers_levelup = [(multi_intersection_combo(liveline_datastream.y_outliers, level = level+1)) for level in range(len(liveline_datastream.y_outliers))]
# # y_outliers_levelup = [y_outliers_levelup_i for y_outliers_levelup_i in y_outliers_levelup if len(y_outliers_levelup_i)>0]
#
#
#
# for y_outliers_levelup_i in y_outliers_levelup:
#     print(len(y_outliers_levelup_i))
#
#
# y_levels = np.arange(liveline_datastream.num_examples)*0
#
# # for level in reversed(range(len(y_outliers_levelup))):
# for level in range(len(liveline_datastream.y_outliers)):
#     print(level)
#     y_levels[y_outliers_levelup[level]] = level
#
# print(np.unique(y_levels,return_counts=True))
#
#


# print(len(y_outliers_levelup))

#
# dataset_folder = "multi-stage-dataset/"
# output_stage=1
# input_stage=1
# upper_quartile= 80
#
#
# lower_quartile = 100 - upper_quartile
# df = pd.read_csv(dataset_folder+"continuous_factory_process.csv")
#
# #drop columns
# df = df.drop(["time_stamp"], axis=1)
# column_names = df.columns
#
# # filter Y columns
# if output_stage == 1:
#     Y_df_actual = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Actual" in name)]]
#     Y_df_setpoint = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Setpoint" in name)]]
#     Y_df_actual.columns =[str(i) for i in range(len(Y_df_actual.columns))]
#     Y_df_setpoint.columns = [str(i) for i in range(len(Y_df_setpoint.columns))]
#     Y_df = (Y_df_setpoint-Y_df_actual).abs()
#
#     # Y_df = df[[name for name in df.columns if ("Stage1.Output" in name) and ("Actual" in name)]]
#
# else:
#     Y_df_actual = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Actual" in name)]]
#     Y_df_setpoint = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Setpoint" in name)]]
#     Y_df_actual.columns =[str(i) for i in range(len(Y_df_actual.columns))]
#     Y_df_setpoint.columns = [str(i) for i in range(len(Y_df_setpoint.columns))]
#     Y_df = (Y_df_setpoint-Y_df_actual).abs()
#
#     # Y_df = df[[name for name in df.columns if ("Stage2.Output" in name) and ("Actual" in name)]]
# # self.Y_df = Y_df
# # filter X columns
# if input_stage ==1:
#     X_df = df[[col_name for col_name in df.columns if ("Machine1" in col_name) or ("Machine2" in col_name) or ("Machine3" in col_name)]]
# elif input_stage ==2:
#     X_df = df[[col_name for col_name in df.columns if ("Machine4" in col_name) or ("Machine5" in col_name)]]
# else:
#     X_df = df[[col_name for col_name in df.columns if ("Machine1" in col_name) or ("Machine2" in col_name) or ("Machine3" in col_name) or ("Machine4" in col_name) or ("Machine5" in col_name)]]
#
# X_columns = X_df.columns
# X = X_df.values
# Y = Y_df.values
#
# plt.figure()
# plt.boxplot([Y[:,i] for i in range(Y.shape[-1])])
#


# plt.boxplot([Y[:,0], Y[:,1]])


# level = 2
# y_outliers_levelup = multi_intersection_combo(liveline_datastream.y_outliers, level = level)
# print(len(y_outliers_levelup))
#
#
