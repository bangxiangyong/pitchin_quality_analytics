import numpy as np

def calc_picp(x_test, y_test_mu, y_test_std, scale):
    within_interval = [len(np.argwhere((x_test[sample_i] <y_test_mu[sample_i] +scale*y_test_std[sample_i]) & (x_test[sample_i] >y_test_mu[sample_i]-scale*y_test_std[sample_i]))) for sample_i in range(x_test.shape[0])]
    return np.array(within_interval)/x_test.shape[-1]

def calc_perc_boundary(nll_mu, nll_train_mu_upper):
    perc_in_boundary = [len(np.argwhere(nll_mu[sample_i]<=nll_train_mu_upper))/len(nll_mu[sample_i]) for sample_i in range(nll_mu.shape[0])]
    return np.array(perc_in_boundary)
