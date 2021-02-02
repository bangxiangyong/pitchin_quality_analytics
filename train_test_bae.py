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

bae_set_seed(55)
input_stage = 1
output_stage = 2
train_model = True
upper_quartile = 80
lower_quartile = 100-upper_quartile
liveline_datastream = Liveline_DataStream(output_stage=output_stage,
                                          input_stage=input_stage,
                                          upper_quartile=upper_quartile)

x_train = liveline_datastream.quantities["train"]
x_test = liveline_datastream.quantities["test"]
x_ood = liveline_datastream.quantities["ood"]
X_columns = liveline_datastream.X_columns
homoscedestic_mode = "none"
likelihood = "gaussian"

#=======AutoEncoder architecture
latent_dim = 100
encoder = Encoder([DenseLayers(architecture=[500,100],
                               output_size=latent_dim,
                               input_size=x_train.shape[-1],
                               activation='leakyrelu',
                               last_activation='leakyrelu')])
decoder_mu = infer_decoder(encoder, last_activation='sigmoid')
decoder_sig = infer_decoder(encoder, last_activation='none')
autoencoder = Autoencoder(encoder=encoder, decoder_mu=decoder_mu, homoscedestic_mode=homoscedestic_mode)
# autoencoder = Autoencoder(encoder=encoder, decoder_mu=decoder_mu, homoscedestic_mode="none")

# autoencoder = Autoencoder(encoder=encoder, decoder_mu=decoder_mu, decoder_sig=decoder_sig)
# autoencoder = Hydra_Autoencoder(encoder=encoder, decoder_sig=decoder_sig)
# autoencoder = Hydra_Autoencoder(encoder=encoder, num_cluster_samples=0, cluster_architecture=[500])

#=========cluster layer=====
batch_size = x_train.shape[0]
num_samples = 10
learning_rate = 0.01

#======training====
num_epoch = 250
bae_model = BAE_Ensemble(autoencoder=autoencoder,
                         num_samples=num_samples, learning_rate=learning_rate, homoscedestic_mode=homoscedestic_mode,
                         likelihood=likelihood, use_cuda=True)

#pretrain to optimise reconstruction loss
if train_model:
    if not bae_model.decoder_sigma_enabled:
        bae_model.fit(x_train, num_epochs=num_epoch, mode="mu")
    else:
        bae_model.fit(x_train, num_epochs=num_epoch, mode="sigma")
    bae_model.model_name = bae_model.model_name +"_"+str(output_stage)
    save_bae_model(bae_model)
else:
    bae_model = load_bae_model("BAE_Ensemble_"+str(output_stage)+".p",folder="trained_models/")

#===================predict==========================
nll_key = "nll_sigma" if bae_model.decoder_sigma_enabled else "nll_homo"
if likelihood == "bernoulli":
    nll_key = "bce"
elif likelihood == "cbernoulli":
    nll_key = "cbce"

ood_args = np.argwhere(liveline_datastream.target["ood"]==11).flatten()
x_ood = liveline_datastream.quantities["ood"][ood_args]

nll_train = bae_model.predict_samples(x_train, select_keys=[nll_key])
nll_test = bae_model.predict_samples(x_test, select_keys=[nll_key])
nll_ood = bae_model.predict_samples(x_ood, select_keys=[nll_key])
y_pred_test = bae_model.predict_samples(x_test, select_keys=["y_mu"])
y_pred_ood = bae_model.predict_samples(x_ood, select_keys=["y_mu"])

nll_train_mu = nll_train.mean(0)[0]
nll_test_mu = nll_test.mean(0)[0]
nll_test_var = nll_test.var(0)[0]
nll_ood_mu = nll_ood.mean(0)[0]
nll_ood_var = nll_ood.var(0)[0]

if bae_model.decoder_sigma_enabled:
    y_test_var = y_pred_test.var(0)[0] + bae_model.predict_samples(x_test, select_keys=["y_sigma"]).mean(0)[0]
    y_ood_var = y_pred_ood.var(0)[0] + bae_model.predict_samples(x_ood, select_keys=["y_sigma"]).mean(0)[0]

elif bae_model.homoscedestic_mode == "every":
    y_test_var = y_pred_test.var(0)[0] + bae_model.get_homoscedestic_noise(return_mean=False)[0]
    y_ood_var = y_pred_ood.var(0)[0] + bae_model.get_homoscedestic_noise(return_mean=False)[0]
else:
    y_test_var = y_pred_test.var(0)[0]
    y_ood_var = y_pred_ood.var(0)[0]

encoded_test = bae_model.predict_latent(x_test, transform_pca=False)
encoded_ood = bae_model.predict_latent(x_ood, transform_pca=False)

#===================evaluate AUROC=============================
max_magnitude = 1000000
auroc_score_nllmu, gmean_nllmu, aps_nllmu, tpr_new_nllmu, fpr_new_nllmu = calc_all_scores(nll_test_mu.mean(-1), nll_ood_mu.mean(-1))
auroc_score_nllvar, gmean_nllvar, aps_nllvar, tpr_new_nllvar, fpr_new_nllvar = calc_all_scores(np.clip(nll_test_var.mean(-1),-max_magnitude,max_magnitude),np.clip(nll_ood_var.mean(-1),-max_magnitude,max_magnitude))
auroc_score_yvar, gmean_yvar, aps_yvar, tpr_new_yvar, fpr_new_yvar = calc_all_scores(np.clip(y_test_var.mean(-1),-max_magnitude,max_magnitude),np.clip(y_ood_var.mean(-1),-max_magnitude,max_magnitude))
auroc_score_enc_var, gmean_enc_var, aps_enc_var, tpr_new_enc_var, fpr_new_enc_var = calc_all_scores(np.sum(encoded_test[1],-1),np.sum(encoded_ood[1],-1))

score_nll_mu = {"auroc":auroc_score_nllmu, "gmean":gmean_nllmu, "aps":aps_nllmu, "tpr_new":tpr_new_nllmu, "fpr_new":fpr_new_nllmu}
score_nll_var = {"auroc":auroc_score_nllvar, "gmean":gmean_nllvar, "aps":aps_nllvar, "tpr_new":tpr_new_nllvar, "fpr_new":fpr_new_nllvar}
score_y_var = {"auroc":auroc_score_yvar, "gmean":gmean_yvar, "aps":aps_yvar, "tpr_new":tpr_new_yvar, "fpr_new":fpr_new_yvar}
score_enc_var = {"auroc":auroc_score_enc_var, "gmean":gmean_enc_var, "aps":aps_enc_var, "tpr_new":tpr_new_enc_var, "fpr_new":fpr_new_enc_var}

df_score = pd.DataFrame([score_nll_mu,score_nll_var,score_y_var,score_enc_var])
df_score.index = ["NLL_MU","NLL_VAR","YVAR","ENCVAR"]

print(df_score)
#=================PLOT TORNADO=================================





# nll_train_mu_upper = np.percentile(nll_train_mu, 75, axis=0)
# nll_train_mu_lower = np.percentile(nll_train_mu, 25, axis=0)
#
# nll_conc = np.concatenate((nll_train_mu,nll_test_mu,nll_ood_mu))

# plot_machines_tornado(nll_test_mu, nll_test_var,nll_train_mu_upper, X_columns, model_stage=model_stage, sample_i=5, figsize=(15,9))
# plot_machines_tornado(nll_ood_mu, nll_ood_var,nll_train_mu_upper,X_columns, model_stage=model_stage, sample_i=-1, figsize=(15,9))

#===================PLOT LATENT SPACE===============================

# fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
# ax1.boxplot([nll_test_mu.mean(-1), nll_ood_mu.mean(-1)])
# ax2.boxplot([np.clip(nll_test_var.mean(-1),-max_magnitude,max_magnitude), np.clip(nll_ood_var.mean(-1),-max_magnitude,max_magnitude)])
# ax3.boxplot([np.clip(y_test_var.mean(-1),-max_magnitude,max_magnitude), np.clip(y_ood_var.mean(-1),-max_magnitude,max_magnitude)])
# ax4.boxplot([np.sum(encoded_test[1],-1), np.sum(encoded_ood[1],-1)])
#
# ax1.set_xticks([1, 2], ['Test', 'OOD'])
# ax2.set_xticks([1, 2], ['Test', 'OOD'])
# ax3.set_xticks([1, 2], ['Test', 'OOD'])
# ax4.set_xticks([1, 2], ['Test', 'OOD'])






