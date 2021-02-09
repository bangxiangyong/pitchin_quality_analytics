import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from agentMET4FOF_ml_extension.agentMET4FOF.agentMET4FOF.streams import DataStreamMET4FOF
from baetorch.baetorch.util.seed import bae_set_seed
from util.calc_outlier import get_num_outliers_df

bae_set_seed(1231)

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

        self.X_columns = X_columns

