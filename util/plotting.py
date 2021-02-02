import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go

# calculate which features are within boundaries
def plot_tornado(nll_mu, nll_var, nll_train_mu_upper, X_columns, sample_i=100, fig=None, ax=None):
    if ax is None and fig is None:
        fig, ax = plt.subplots()

    # Example data
    num_features = len(X_columns)
    y_pos = np.arange(num_features)

    in_boundary = np.argwhere(nll_mu[sample_i]<=nll_train_mu_upper)
    contrib_color = np.array(["green" if i in in_boundary else "red" for i in range(num_features)])

    performance = nll_mu[sample_i]
    error = nll_var[sample_i]**0.5

    ax.barh(y_pos, -1*performance, xerr=error, align='center', capsize=3, color =contrib_color, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(X_columns)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Log likelihood contribution')
    # ax.set_title('Tornado plot for sample #%s' %str(sample_i))

    ax.scatter(-1*nll_train_mu_upper,np.arange(num_features), s=0.35, color="red", alpha=0.8)

    return fig

# multiple tornado plots
def get_machine_col_args(X_columns, machine_num=1):
    machine_col_names = [col_name for col_name in X_columns if "Machine"+str(machine_num) in col_name]
    machine_col_args = [np.argwhere(X_columns == machine_col_i)[0][0] for machine_col_i in machine_col_names]
    return machine_col_names, machine_col_args

def plot_machines_tornado(nll_mu, nll_var, nll_train_mu_upper, X_columns, model_stage=1, sample_i=100, figsize=(10,9)):
    machine_nums = [1,2,3] if model_stage == 1 else [4,5]
    # fig, axes = plt.subplots(len(machine_nums), 1)
    fig, axes = plt.subplots(1,len(machine_nums), figsize=figsize)
    axes = axes.flatten()
    for machine_num,ax in zip(machine_nums,axes):
        machine_col_names, machine_col_args = get_machine_col_args(X_columns, machine_num=machine_num)
        cleaned_x_col = [x_col.split('.')[1]+x_col.split('.')[2] if "RawMaterial" in x_col else x_col.split('.')[1] for x_col in X_columns[machine_col_args]]
        fig = plot_tornado(nll_mu[:,machine_col_args], nll_var[:,machine_col_args], nll_train_mu_upper[machine_col_args], sample_i=sample_i, X_columns=cleaned_x_col, fig=fig,ax=ax)
        ax.set_title("%s" % ("Machine "+str(machine_num)))
    fig.tight_layout()
    return fig

def plot_total_nll(data, sender_agent, xname='Time',yname='Y'):
    """
    Parameters
    ----------
    data : dict or np.darray
        The data saved in the MonitorAgent's memory, for each Inputs (Agents) it is connected to.

    sender_agent : str
        Name of the sender agent

    **kwargs
        Custom parameters.
        In this example, xname and yname  are the keys of the data in the Monitor agent's memory.
    """

    y = data[:,0]*-1
    x = np.arange(len(y))
    y_unc = data[:,1]

    trace = go.Scatter(x=x, y=y,mode="lines", name=sender_agent)

    y_upper = y +2* y_unc
    y_lower = y -2* y_unc

    unc_trace = go.Scatter(
        x=list(x)+list(x[::-1]), # x, then x reversed
        y=list(y_upper)+list(y_lower[::-1]), # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.25)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,

    )


    return [trace,unc_trace]
