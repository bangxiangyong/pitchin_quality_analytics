from agentMET4FOF_ml_extension.agentMET4FOF.agentMET4FOF.agents import AgentMET4FOF
from agentMET4FOF_ml_extension.agentMET4FOF.agentMET4FOF.streams import DataStreamMET4FOF
from agentMET4FOF_ml_extension.agentMET4FOF_ml_extension.agents import ML_DatastreamAgent, ML_TransformAgent, ML_EvaluateAgent
from baetorch.baetorch.util.seed import bae_set_seed
from multi_stage_dataset import Liveline_DataStream

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

from multi_stage_dataset import Liveline_DataStream
from util.calc_auroc import calc_auroc_score, calc_all_scores
from util.calc_picp import calc_picp, calc_perc_boundary
from util.plotting import plot_tornado, plot_machines_tornado
import pandas as pd


# Datastream Agent
class Liveline_DatastreamAgent(ML_DatastreamAgent):
    """
    A base class for ML data-streaming agent, which takes into account the train/test split.

    The agent_loop behaviour are based on "current_state":
    1) in "Train", it will send out train and test batch of data, and then set to "Idle" state
    2) in "Simulate", it will send out test data one-by-one in a datastream fashion.
    3) Output form : {"quantities": iterable, "target": iterable}
    """

    parameter_choices = {"input_stage": [1,2], "target_stage": [1,2], "train_size": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
    # parameter_map = {"input_stage":{"IRIS":IRIS_Datastream}}

    def init_parameters(self, random_state=123, input_stage=1, target_stage=1, train_size=0.5, simulate_batch_size=200):
        self.input_stage = input_stage
        self.target_stage = target_stage
        self.train_size = train_size
        self.random_state = random_state
        bae_set_seed(random_state)
        liveline_datastream = Liveline_DataStream(output_stage=target_stage,
                                                  input_stage=input_stage,
                                                  train_size=train_size)
        self.datastream = liveline_datastream
        self.x_train = liveline_datastream.quantities["train"]
        self.x_test = liveline_datastream.quantities["test"]
        self.x_ood = liveline_datastream.quantities["ood"]

        self.y_train = liveline_datastream.quantities["train"]
        self.y_test = liveline_datastream.quantities["test"]
        self.y_ood = liveline_datastream.quantities["ood"]

        # for simulation
        self.datastream_simulate = DataStreamMET4FOF()
        self.datastream_simulate.set_data_source(quantities=np.concatenate((self.x_test,self.x_ood)))
        self.simulate_batch_size = simulate_batch_size

    def agent_loop(self):
        if self.current_state == "Running":
            self.send_output({"quantities":self.x_train, "target":self.y_train},channel="train")
            self.send_output({"quantities":{"test":self.x_test,"ood":self.x_ood},
                              "target":{"test":self.y_test,"ood":self.y_ood}},
                             channel="test")
            # self.current_state = "Idle"
            self.current_state = "Simulate"

        elif self.current_state == "Simulate":
            self.send_output({"quantities": self.datastream_simulate.next_sample(batch_size=self.simulate_batch_size)["quantities"],
                              "metadata":self.datastream.X_columns},
                             channel="simulate")


# BAE Agent
class BAE_Agent(ML_TransformAgent):
    """
    Fully connected BAE Agent.
    """

    parameter_choices = {"first_nodes":[0,100,200,300,400,500],
                         "second_nodes":[0,100,200,300,400,500],
                         "latent_dim": [10,50,100],
                         "likelihood":["1_gaussian", "homo_gaussian", "hetero_gaussian", "bernoulli", "cbernoulli"],
                         "learning_rate":[0.1,0.01,0.001],
                         "bae_samples":[1,5,10],
                         # "use_cuda": [True,False],
                         # "train_model":[True,False],
                         "num_epochs":[10,50,100,150,200,250]
                         }

    def init_parameters(self, first_nodes=500,
                        second_nodes=100,
                        latent_dim=100,
                        likelihood="1_gaussian",
                        learning_rate =0.01,
                        bae_samples=5,
                        use_cuda=True,
                        train_model=True,
                        num_epochs=250,
                        **data_params):
        self.first_nodes = first_nodes
        self.second_nodes = second_nodes
        self.latent_dim = latent_dim
        self.architecture = [nodes for nodes in [self.first_nodes, self.second_nodes] if nodes >0]

        self.likelihood = likelihood
        self.learning_rate = learning_rate
        self.homoscedestic_mode_map = {"1_gaussian":"none", "homo_gaussian":"every",
                                  "hetero_gaussian":"none",
                                  "bernoulli":"none",
                                  "cbernoulli":"none"}
        self.likelihood_mode_map = {"1_gaussian":"gaussian", "homo_gaussian":"gaussian",
                                  "hetero_gaussian":"gaussian",
                                  "bernoulli":"bernoulli",
                                  "cbernoulli":"cbernoulli"}
        self.bae_samples = bae_samples
        self.use_cuda = use_cuda
        self.train_model = train_model
        self.num_epochs= num_epochs

    def fit(self, message_data):
        x_train = message_data["quantities"]

        # =======AutoEncoder architecture
        encoder = Encoder([DenseLayers(architecture=self.architecture,
                                       output_size=self.latent_dim,
                                       input_size=x_train.shape[-1],
                                       activation='leakyrelu',
                                       last_activation='leakyrelu')])
        decoder_mu = infer_decoder(encoder, last_activation='sigmoid')
        decoder_sig = infer_decoder(encoder, last_activation='none')
        if self.likelihood != "hetero_gaussian":
            autoencoder = Autoencoder(encoder=encoder, decoder_mu=decoder_mu, homoscedestic_mode=self.homoscedestic_mode_map[self.likelihood])
        else:
            autoencoder = Autoencoder(encoder=encoder, decoder_mu=decoder_mu, decoder_sig=decoder_sig)

        # ======training====
        bae_model = BAE_Ensemble(autoencoder=autoencoder,
                                 num_samples=self.bae_samples, learning_rate=self.learning_rate,
                                 homoscedestic_mode=self.homoscedestic_mode_map[self.likelihood],
                                 likelihood=self.likelihood_mode_map[self.likelihood], use_cuda=self.use_cuda)

        # pretrain to optimise reconstruction loss
        if self.train_model:
            if not bae_model.decoder_sigma_enabled:
                bae_model.fit(x_train, num_epochs=self.num_epochs, mode="mu")
            else:
                bae_model.fit(x_train, num_epochs=self.num_epochs, mode="sigma")
            bae_model.model_name = bae_model.model_name + "_" + self.name
            save_bae_model(bae_model)
            self.bae_model = bae_model
        else:
            bae_model = load_bae_model("BAE_Ensemble_" + self.name + ".p", folder="trained_models/")
            self.bae_model = bae_model

    def transform(self, message_data):
        """
        Apply transform on every key of the quantities' dict if available
        Dict can have keys such as "test" , "ood", "train"
        Otherwise, directly compute on the quantities (assumed to be iterable)

        Each transformed data samples have a dict of {"nll_mu":nll_test_mu, "nll_var": nll_test_var, "y_var":y_test_var, "enc_var":encoded_test}

        e.g message_data["quantities"]["test"]["nll_mu"] OR message_data["quantities"]["nll_mu"]

        """

        if isinstance(message_data["quantities"],dict):
            return {key:self.transform_(message_data["quantities"][key]) for key in message_data["quantities"].keys()}
        else:
            x_test = message_data["quantities"]
            y_pred_test = self.transform_(x_test)
            return y_pred_test

    def transform_(self, x_test):
        """
        Internal transform function to compute NLL-MU, NLL-VAR, Y-PRED-VAR, and ENCODED-VAR
        """

        bae_model = self.bae_model
        # ===================predict==========================
        nll_key = "nll_sigma" if bae_model.decoder_sigma_enabled else "nll_homo"
        if self.likelihood == "bernoulli":
            nll_key = "bce"
        elif self.likelihood == "cbernoulli":
            nll_key = "cbce"

        nll_test = bae_model.predict_samples(x_test, select_keys=[nll_key])
        y_pred_test = bae_model.predict_samples(x_test, select_keys=["y_mu"])

        # compute statistics over BAE sampled parameters
        nll_test_mu = nll_test.mean(0)[0]
        nll_test_var = nll_test.var(0)[0]

        # get predictive uncertainty
        if bae_model.decoder_sigma_enabled:
            y_test_var = y_pred_test.var(0)[0] + bae_model.predict_samples(x_test, select_keys=["y_sigma"]).mean(0)[0]
        elif bae_model.homoscedestic_mode == "every":
            y_test_var = y_pred_test.var(0)[0] + bae_model.get_homoscedestic_noise(return_mean=False)[0]
        else:
            y_test_var = y_pred_test.var(0)[0]

        # get encoded data
        encoded_test = bae_model.predict_latent(x_test, transform_pca=False)

        return {"nll_mu":nll_test_mu, "nll_var": nll_test_var, "y_var":y_test_var, "enc_var":encoded_test}

class OOD_EvaluateAgent(ML_EvaluateAgent):
    """
    Last piece in the ML-Pipeline to evaluate the model's performance on the datastream.

    If ml_experiment_proxy is specified, this agent will save the results upon finishing.
    """
    parameter_choices = {}

    def init_parameters(self, evaluate_method=[], ml_experiment_proxy=None, **evaluate_params):
        self.ml_experiment_proxy = ml_experiment_proxy

    def on_received_message(self, message):

        if message["channel"] == "test":
            message_data_quantities = message["data"]["quantities"]
            nll_test_mu = message_data_quantities["test"]["nll_mu"]
            nll_test_var = message_data_quantities["test"]["nll_var"]
            y_test_var = message_data_quantities["test"]["y_var"]
            enc_test_var = message_data_quantities["test"]["enc_var"]

            nll_ood_mu = message_data_quantities["ood"]["nll_mu"]
            nll_ood_var = message_data_quantities["ood"]["nll_var"]
            y_ood_var = message_data_quantities["ood"]["y_var"]
            enc_ood_var = message_data_quantities["ood"]["enc_var"]

            max_magnitude = 1000000
            auroc_score_nllmu, gmean_nllmu, aps_nllmu, tpr_new_nllmu, fpr_new_nllmu = calc_all_scores(nll_test_mu.mean(-1), nll_ood_mu.mean(-1))
            auroc_score_nllvar, gmean_nllvar, aps_nllvar, tpr_new_nllvar, fpr_new_nllvar = calc_all_scores(
                np.clip(nll_test_var.mean(-1), -max_magnitude, max_magnitude),
                np.clip(nll_ood_var.mean(-1), -max_magnitude, max_magnitude))
            auroc_score_yvar, gmean_yvar, aps_yvar, tpr_new_yvar, fpr_new_yvar = calc_all_scores(
                np.clip(y_test_var.mean(-1), -max_magnitude, max_magnitude),
                np.clip(y_ood_var.mean(-1), -max_magnitude, max_magnitude))
            auroc_score_enc_var, gmean_enc_var, aps_enc_var, tpr_new_enc_var, fpr_new_enc_var = calc_all_scores(
                np.sum(enc_test_var[1], -1), np.sum(enc_ood_var[1], -1))

            score_nll_mu = {"auroc": auroc_score_nllmu, "gmean": gmean_nllmu, "aps": aps_nllmu,
                            "tpr": tpr_new_nllmu, "fpr": fpr_new_nllmu}
            score_nll_var = {"auroc": auroc_score_nllvar, "gmean": gmean_nllvar, "aps": aps_nllvar,
                             "tpr": tpr_new_nllvar, "fpr": fpr_new_nllvar}
            score_y_var = {"auroc": auroc_score_yvar, "gmean": gmean_yvar, "aps": aps_yvar, "tpr": tpr_new_yvar,
                           "fpr": fpr_new_yvar}
            score_enc_var = {"auroc": auroc_score_enc_var, "gmean": gmean_enc_var, "aps": aps_enc_var,
                             "tpr": tpr_new_enc_var, "fpr": fpr_new_enc_var}

            score_nll_mu = {key+"-nll-mu":val for key,val in score_nll_mu.items()}
            score_nll_var = {key + "-nll-var": val for key, val in score_nll_var.items()}
            score_y_var = {key + "-y-var": val for key, val in score_y_var.items()}
            score_enc_var = {key + "-enc-var": val for key, val in score_enc_var.items()}
            results ={}
            for result in [score_nll_mu,score_nll_var, score_y_var, score_enc_var]:
                results.update(result)

            self.log_info(str(results))
            self.upload_result(results)

class GeneratePlotAgent(AgentMET4FOF):
    """
    Plots BAE outputs in simulate mode.

    1. Extracts metadata for plotting
    2. in train channel, it determines the NLL_TRAIN_MU_UPPER_BOUND
    3. in simulate channel, it plots the tornado chart (for explainability)
    and sends a tuple to monitor agent for uncertainty plot

    """
    parameter_choices = {"model_stage":1}

    def init_parameters(self, figsize=(12,5), model_stage=1):
        self.figsize= figsize
        self.model_stage = model_stage

    def on_received_message(self, message):
        if isinstance(message["data"], dict) and "metadata" in message["data"].keys():
            self.metadata = message["data"]["metadata"]

        if message["channel"] == "train":
            nll_train_mu = message["data"]["quantities"]["nll_mu"]
            self.nll_train_mu_upper = np.percentile(nll_train_mu, 95, axis=0)

        if message["channel"] == "simulate":
            nll_test_mu = message["data"]["quantities"]["nll_mu"]
            nll_test_var = message["data"]["quantities"]["nll_var"]
            fig = self.plot_tornado(nll_test_mu, nll_test_var)
            self.send_plot(fig, mode="image")
            output_send = np.moveaxis(np.array((nll_test_mu.mean(-1),(nll_test_var.mean(-1)**0.5))),0,1)
            self.log_info(str(output_send.shape))
            self.send_output(output_send)

    def plot_tornado(self, nll_test_mu, nll_test_var):
        fig = plot_machines_tornado(nll_test_mu, nll_test_var, self.nll_train_mu_upper, sample_i=-1,
                                    X_columns=self.metadata, figsize=self.figsize,model_stage=self.model_stage)

        return fig














