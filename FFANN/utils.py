import torch
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats as stats
from pathlib import Path
from captum.attr import IntegratedGradients
import torch.nn.functional as f
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchinfo import summary
import logging
import pandas as pd
import data, models

def get_file_dir():
    """[summary]

    :return: [description]
    :rtype: [type]
    """    
    return Path(__file__).resolve().parent

def get_data_dir():
    """[summary]

    :return: [description]
    :rtype: [type]
    """    
    return get_file_dir().parent / 'data'

def get_dims(dataset, prob=False):
    """[summary]

    :param dataset: [description]
    :type dataset: [type]
    :param prob: [description], defaults to False
    :type prob: bool, optional
    :return: [description]
    :rtype: [type]
    """    
    x, y = dataset[0]
    in_dim = torch.numel(x)
    out_dim = 2 * torch.numel(y) if prob else torch.numel(y)
    return in_dim, out_dim

def get_data(data_loader, device='cpu'):
    """[summary]

    :param data_loader: [description]
    :type data_loader: [type]
    :param device: [description], defaults to 'cpu'
    :type device: str, optional
    :return: [description]
    :rtype: [type]
    """    
    x_list, y_list = [], []
    for x_batch, y_batch in list(data_loader):
        x_list.append(x_batch)
        y_list.append(y_batch)
    x = torch.cat(x_list)
    y = torch.cat(y_list)
    return x.to(device), y.to(device)

def plot_data(axs, x_dict, y_dict, train_x, train_y, val_x, val_y):
    """[summary]

    :param axs: [description]
    :type axs: [type]
    :param x_dict: [description]
    :type x_dict: [type]
    :param y_dict: [description]
    :type y_dict: [type]
    :param train_x: [description]
    :type train_x: [type]
    :param train_y: [description]
    :type train_y: [type]
    :param val_x: [description]
    :type val_x: [type]
    :param val_y: [description]
    :type val_y: [type]
    """    
    train_samples, val_samples = train_x.size(0), val_x.size(0)
    for xi, x in enumerate(x_dict):
        for yi, y in enumerate(y_dict):
            axs[yi,xi].scatter(train_x[:,x['idx']], train_y[:,y['idx']], c='r', s=1, label=f'training data ({train_samples} samples)')
            axs[yi,xi].scatter(val_x[:,x['idx']], val_y[:,y['idx']], c='b', s=1, label=f'validation data ({val_samples} samples)')
            axs[yi,xi].set_xlabel(x['name'])
            axs[yi,xi].set_ylabel(y['name'])

def plot_test_data(axs, x_dict, y_dict, test_x, test_y):
    """[summary]

    :param axs: [description]
    :type axs: [type]
    :param x_dict: [description]
    :type x_dict: [type]
    :param y_dict: [description]
    :type y_dict: [type]
    :param test_x: [description]
    :type test_x: [type]
    :param test_y: [description]
    :type test_y: [type]
    """    
    test_samples = test_x.size(0)
    for xi, x in enumerate(x_dict):
        for yi, y in enumerate(y_dict):
            axs[yi,xi].scatter(test_x[:,x['idx']], test_y[:,y['idx']], c='g', s=1, label=f'test data ({test_samples} samples)')
            axs[yi,xi].set_xlabel(x['name'])
            axs[yi,xi].set_ylabel(y['name'])

def model_training(model, loss_fn, optimizer, train_loader, val_loader, epochs, verbose=False):
    """[summary]

    :param model: [description]
    :type model: [type]
    :param loss_fn: [description]
    :type loss_fn: [type]
    :param optimizer: [description]
    :type optimizer: [type]
    :param train_loader: [description]
    :type train_loader: [type]
    :param val_loader: [description]
    :type val_loader: [type]
    :param epochs: [description]
    :type epochs: [type]
    :param verbose: [description], defaults to False
    :type verbose: bool, optional
    :raises Exception: [description]
    :return: [description]
    :rtype: [type]
    """    
    early_stop_patience = 1
    early_stop_counter = early_stop_patience
    epoch_list = []
    train_losses = []
    val_losses = []
    best_epoch = 0
    best_loss = float('inf')
    best_parameters = model.state_dict()
    for t in range(epochs):
        epoch_list.append(t+1)
        # training step:
        model.training_step(train_loader, loss_fn, optimizer)
        train_loss = model.loss_calculation(train_loader, loss_fn)
        train_losses.append(train_loss)
        if np.isnan(train_loss):
            raise Exception('training loss is not a number')
        # validation step:
        val_loss = model.loss_calculation(val_loader, loss_fn)
        val_losses.append(val_loss)
        # early stopping:
        if t > int(0.1 * epochs) and val_loss < best_loss:
            if early_stop_counter < early_stop_patience:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
                best_epoch, best_loss = t, val_loss
                best_parameters = model.state_dict()
        # status update:
        if verbose and ((t+1) % 1000 == 0):
            print(f"Epoch {t+1}: training loss {train_loss:>8f}, validation loss {val_loss:>8f}")
    model.load_state_dict(best_parameters)
    return train_losses, val_losses, best_epoch

def plot_training_history(ax, train_losses, val_losses, best_epoch):
    """[summary]

    :param train_losses: [description]
    :type train_losses: [type]
    :param val_losses: [description]
    :type val_losses: [type]
    :param best_epoch: [description]
    :type best_epoch: [type]
    """    
    epoch_list = torch.arange(len(train_losses))+1
    if min(train_losses) < 0:
        # probabilistic loss
        ax.plot(epoch_list, train_losses, linestyle='solid', alpha=1, label='prob. training loss')
        ax.plot(epoch_list, val_losses, linestyle='solid', alpha=0.7, label='prob. validation loss')
        #plt.ylim(top=0, bottom=-60)
        print(f'Best epoch ({best_epoch}): prob. training loss {train_losses[best_epoch]}, prob. validation loss {val_losses[best_epoch]}')
    else:
        # deterministic loss
        ax.semilogy(epoch_list, train_losses, linestyle='solid', alpha=1, label='training loss')
        ax.semilogy(epoch_list, val_losses, linestyle='solid', alpha=0.7, label='validation loss')
        print(f'Best epoch ({best_epoch}): training loss {train_losses[best_epoch]:e}, validation loss {val_losses[best_epoch]:e}')
    ax.axvline(x=best_epoch, color='k', linestyle='dashed', label='best epoch')
    ax.legend()
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')

def plot_predictions(ax, y, pred, idx=0, title=None, label=None, legend=True):
    """[summary]

    :param ax: [description]
    :type ax: [type]
    :param y: [description]
    :type y: [type]
    :param pred: [description]
    :type pred: [type]
    :param idx: [description], defaults to 0
    :type idx: int, optional
    :param title: [description], defaults to None
    :type title: [type], optional
    :param label: [description], defaults to None
    :type label: [type], optional
    """    
    indices = torch.arange(y.size(0))
    y_entry, pred_entry = y[:,idx], pred[:,idx]
    order = torch.argsort(pred_entry)
    y_sorted, mu_sorted = y_entry[order], pred_entry[order]
    ax.scatter(indices, y_sorted, color='k', marker='.', s=2, label=r'$y$')
    ax.plot(indices, mu_sorted, 'b', label=r'$\hat{y}$')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('indices [-]')
    if label is not None:
        ax.set_ylabel(label)
    if legend:
        ax.legend()
    ax.set_axisbelow(True)
    ax.grid()

def plot_confidence_intv(ax, y, pred, idx=0, sort=0, scales=None, title=None, label=None, legend=True, marker='.'):
    """[summary]

    :param ax: [description]
    :type ax: [type]
    :param y: [description]
    :type y: [type]
    :param pred: [description]
    :type pred: [type]
    :param idx: [description], defaults to 0
    :type idx: int, optional
    :param scales: [description], defaults to None
    :type scales: [type], optional
    :param title: [description], defaults to None
    :type title: [type], optional
    :param label: [description], defaults to None
    :type label: [type], optional
    """    
    indices = torch.arange(y.size(0))
    mu, sigma = torch.tensor_split(pred, 2, dim=1)
    y_entry, mu_entry, sigma_entry = y[:,idx], mu[:,idx], sigma[:,idx]
    if sort == 0:
        order = torch.argsort(mu_entry)
    elif sort == 1:
        order = torch.argsort(sigma_entry)
    elif sort == 2:
        order = torch.argsort(y_entry)
    if sort is None:
        y_sorted, mu_sorted, sigma_sorted = y_entry, mu_entry, sigma_entry
    else:
        y_sorted, mu_sorted, sigma_sorted = y_entry[order], mu_entry[order], sigma_entry[order]
    if scales is None:
        scales = [1,2]
    for i, scale in enumerate(scales):
        lower, upper = mu_sorted - scale * sigma_sorted, mu_sorted + scale * sigma_sorted
        ax.fill_between(indices, lower, upper, color='g', alpha=0.4/(i+1), label=fr'$\mu \pm {scale}\sigma$')
    if sort == 2:
        ax.scatter(indices, mu_sorted, color='b', marker=marker, s=2, label=r'$\mu$')
        ax.plot(indices, y_sorted, 'k', label=r'$y$')
    else:
        ax.scatter(indices, y_sorted, color='k', marker=marker, s=2, label=r'$y$')
        ax.plot(indices, mu_sorted, 'b', label=r'$\mu$')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(r'indices [-]')
    ax.set_ylabel(r'Latent variable $y$ [-]')
    if label is not None:
        ax.set_ylabel(label)
    if legend:
        ax.legend()
    ax.set_axisbelow(True)
    ax.grid()
    return order if sort is not None else None

def reconstruct_kappa(test_y, test_pred, prob=True):
    K = test_y[:,27:33]
    K_mu = test_pred[:,27:33]
    #print(f'{torch.sqrt(torch.max(K_sigma)**2)}')
    Kappa = data.Dataset.reverse_cholesky(K)
    Kappa_mu = data.Dataset.reverse_cholesky(K_mu)
    Kappa_E = Kappa_mu

    if prob:
        K_sigma = test_pred[:,62:68]
        Kappa_sigma = np.zeros_like(Kappa_mu)
        for i in range(Kappa_sigma.shape[0]):
            Kappa_sigma[i,:,:] = np.diag([K_sigma[i,0]**2, K_sigma[i,1]**2+K_sigma[i,2]**2, K_sigma[i,3]**2+K_sigma[i,4]**2+K_sigma[i,5]**2])
        Kappa_E += Kappa_sigma

        Kappa_Var = np.zeros_like(Kappa_mu)
        for i in range(Kappa_Var.shape[0]):
            Kappa_Var[i,:,:] = np.array([[4*K_mu[i,0]**2*K_sigma[i,0]**2, 0, 0],
                                        [K_mu[i,0]**2*K_sigma[i,1]**2+K_mu[i,1]**2*K_sigma[i,0]**2+K_sigma[i,0]**2*K_sigma[i,1]**2,
                                        4*(K_mu[i,1]**2*K_sigma[i,1]**2+K_mu[i,2]**2*K_sigma[i,2]**2), 0],
                                        [K_mu[i,0]**2*K_sigma[i,3]**2+K_mu[i,3]**2*K_sigma[i,0]**2+K_sigma[i,0]**2*K_sigma[i,3]**2,
                                        K_mu[i,1]**2*K_sigma[i,3]**2+K_mu[i,3]**2*K_sigma[i,1]**2+K_mu[i,2]**2*K_sigma[i,4]**2+K_mu[i,4]**2*K_sigma[i,2]**2+K_sigma[i,1]**2*K_sigma[i,3]**2+K_sigma[i,2]**2*K_sigma[i,4]**2,
                                        4*(K_mu[i,3]**2*K_sigma[i,3]**2+K_mu[i,4]**2*K_sigma[i,4]**2+K_mu[i,5]**2*K_sigma[i,5]**2)]])
            Kappa_Var[i,0,1], Kappa_Var[i,0,2], Kappa_Var[i,1,2] = Kappa_Var[i,1,0], Kappa_Var[i,2,0], Kappa_Var[i,2,1]
            Kappa_Var[i,:,:] += np.diag([2*K_sigma[i,0]**4, 2*(K_sigma[i,1]**4+K_sigma[i,2]**4), 2*(K_sigma[i,3]**4+K_sigma[i,4]**4+K_sigma[i,5]**4)])

    def get_entries(Kappa):
        return torch.Tensor(np.vstack((Kappa[:,0,0], Kappa[:,1,1], Kappa[:,2,2], Kappa[:,0,1], Kappa[:,0,2], Kappa[:,1,2])).T)

    Kappa_entries = get_entries(Kappa)
    Kappa_E_entries = get_entries(Kappa_E)
    if prob:
        Kappa_Var_entries = get_entries(Kappa_Var)
        Kappa_pred = torch.hstack((Kappa_E_entries, torch.sqrt(Kappa_Var_entries)))
    else:
        Kappa_pred = Kappa_E_entries
    return Kappa_entries, Kappa_pred


def reconstruct_alpha(test_y, test_pred, prob=True):
    Alpha = test_y[:,21:27]
    Alpha_mu = test_pred[:,21:27]
    Alpha[:,3:5] = Alpha[:,3:5] / np.sqrt(2)
    Alpha_mu[:,3:5] = Alpha_mu[:,3:5] / np.sqrt(2)
    if prob:
        Alpha_sigma = test_pred[:,56:62]
        Alpha_sigma[:,3:5] = Alpha_sigma[:,3:5] / np.sqrt(2)
        Alpha_pred = torch.hstack((Alpha_mu, Alpha_sigma))
    else:
        Alpha_pred = Alpha_mu
    return Alpha, Alpha_pred

def plot_eigenvalues(ax, K, K_mu, K_sigma=None, idx=0, scales=None, p=None, title=None, label=None):
    """[summary]

    :param ax: [description]
    :type ax: [type]
    :param K: [description]
    :type K: [type]
    :param K_mu: [description]
    :type K_mu: [type]
    :param K_sigma: [description], defaults to None
    :type K_sigma: [type], optional
    :param idx: [description], defaults to 0
    :type idx: int, optional
    :param scales: [description], defaults to None
    :type scales: [type], optional
    :param p: [description], defaults to None
    :type p: [type], optional
    :param title: [description], defaults to None
    :type title: [type], optional
    :param label: [description], defaults to None
    :type label: [type], optional
    """    
    Kappa = data.Dataset.reverse_cholesky(K)
    Kappa_mu = data.Dataset.reverse_cholesky(K_mu)
    eig = torch.Tensor(np.linalg.eig(Kappa)[0])
    eig_mu = torch.Tensor(np.linalg.eig(Kappa_mu)[0])
    y_entry, mu_entry = eig[:,idx], eig_mu[:,idx]
    #K1 = data.Dataset.reverse_cholesky(K_mu, K_sigma)
    #eig1 = torch.Tensor(np.linalg.eig(K1)[0])
    #K2 = data.Dataset.reverse_cholesky(K_sigma, K_mu)
    #eig2 = torch.Tensor(np.linalg.eig(K2)[0])
    #print(torch.min(eig1), torch.min(eig2))
    indices = torch.arange(eig.size(0))
    order = torch.argsort(mu_entry)
    y_sorted, mu_sorted = y_entry[order], mu_entry[order]
    if scales is None:
        scales = []
    if not isinstance(scales, list):
        scales = [scales]
    if p is None:
        p = 2
    N = K.shape[0]
    for i, scale in enumerate(scales):
        eig_lower, eig_upper = torch.zeros(p, N, 3), torch.zeros(p, N, 3)
        s_values = torch.linspace(0, scale, steps=p)
        #print(s_values)
        for j, s in enumerate(s_values):
            lower, upper = K_mu - s*K_sigma, K_mu + s*K_sigma
            Kappa_lower = data.Dataset.reverse_cholesky(lower)
            Kappa_upper = data.Dataset.reverse_cholesky(upper)
            eig_lower[j,:] = torch.Tensor(np.linalg.eig(Kappa_lower)[0])
            eig_upper[j,:] = torch.Tensor(np.linalg.eig(Kappa_upper)[0])
        eig_lower = torch.min(eig_lower, dim=0)[0]
        eig_upper = torch.max(eig_upper, dim=0)[0]
        lower_entry, upper_entry = eig_lower[:,idx], eig_upper[:,idx]
        lower_entry, upper_entry = lower_entry[order], upper_entry[order]
        ax.fill_between(indices, lower_entry, upper_entry, color='g', alpha=0.5/(i+1), label=fr'$\mu \pm {scale}\sigma$')
    ax.plot(indices, mu_sorted, label=r'$\mu$')
    ax.scatter(indices, y_sorted, color='grey', marker='x', label=r'$y$')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(r'indices [-]')
    if label is not None:
        ax.set_ylabel(label)
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid()

def plot_ecdf(ax, y, pred, idx=0, title=None, label=None, x_max=6):
    """[summary]

    :param ax: [description]
    :type ax: [type]
    :param y: [description]
    :type y: [type]
    :param pred: [description]
    :type pred: [type]
    :param idx: [description], defaults to 0
    :type idx: int, optional
    :param title: [description], defaults to None
    :type title: [type], optional
    :param label: [description], defaults to None
    :type label: [type], optional
    :param x_max: [description], defaults to 6
    :type x_max: int, optional
    """    
    mu, sigma = torch.tensor_split(pred, 2, dim=1)
    y_entry, mu_entry, sigma_entry = y[:,idx], mu[:,idx], sigma[:,idx]
    distance = torch.abs(y_entry - mu_entry) / sigma_entry
    ecdf = ECDF(distance)
    norm_x = torch.linspace(0, x_max, 100)
    norm_cdf = stats.halfnorm.cdf(norm_x, 0, 1)
    # Calculate L2 norm:
    x_a, x_b = ecdf.x[1:-2], ecdf.x[2:-1]
    emp_y = ecdf.y[1:-2]
    ref_y = stats.halfnorm.cdf((x_a + x_b)/2)
    L2_norm = np.sqrt(np.sum((emp_y - ref_y)**2*(x_b-x_a)))
    ax.plot(ecdf.x, ecdf.y, label='empirical CDF')
    ax.plot(norm_x, norm_cdf, linestyle='dashed', label='reference CDF')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(r'$|y-\mu| / \sigma$')
    ax.set_ylabel(r'cumulative probability [-]')
    if label is None:
        label = r'cumulative probability [-]'
    ax.set_ylabel(label)
    ax.text(3.5,.5,r'$||\mathrm{e.} - \mathrm{r.}||_{L^2} = '+f'{L2_norm:.4f}'+r'$',
        fontsize='large',
        bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},
        ha='center', va='center') 
    ax.set_xlim(0, x_max)
    ax.set_axisbelow(True)
    ax.grid()
    ax.legend()

def error_barplot(error, title=''):
    """[summary]

    :param error: [description]
    :type error: [type]
    :param title: [description], defaults to ''
    :type title: str, optional
    """    
    mean_error = torch.mean(error, dim=0)
    max_error = torch.max(error, dim=0).values
    x_pos = np.arange(len(mean_error))
    width = 0.35
    fig, ax = plt.subplots(figsize=(15,5), dpi=80)
    plot_mean = ax.bar(x_pos - width/2, mean_error, width, label='mean error')
    plot_max = ax.bar(x_pos + width/2, max_error, width, label='max error')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_pos)
    ax.set_yscale('log')
    ax.set_ylim([1e-4,1e-0])
    ax.set_title(title)
    ax.set_xlabel('output features')
    ax.legend()
    fig.tight_layout()
    plt.show()

def MS_losses(test_pred, test_y):
    """[summary]

    :param test_pred: [description]
    :type test_pred: [type]
    :param test_y: [description]
    :type test_y: [type]
    :return: [description]
    :rtype: [type]
    """    
    total_loss = models.MSModel.total_loss(test_pred, test_y)
    L_loss = models.MSModel.stiffness_loss(test_pred, test_y)
    A_loss = models.MSModel.thermal_exp_loss(test_pred, test_y)
    K_loss = models.MSModel.conductivity_loss(test_pred, test_y)
    c_loss = models.MSModel.heat_capacity_loss(test_pred, test_y)
    rho_loss = models.MSModel.density_loss(test_pred, test_y)
    return total_loss, L_loss, A_loss, K_loss, c_loss, rho_loss

def PMS_losses(test_pred, test_y):
    """[summary]

    :param test_pred: [description]
    :type test_pred: [type]
    :param test_y: [description]
    :type test_y: [type]
    :return: [description]
    :rtype: [type]
    """    
    total_loss = models.PMSModel.total_loss(test_pred, test_y)
    L_loss = models.PMSModel.stiffness_loss(test_pred, test_y)
    A_loss = models.PMSModel.thermal_exp_loss(test_pred, test_y)
    K_loss = models.PMSModel.conductivity_loss(test_pred, test_y)
    c_loss = models.PMSModel.heat_capacity_loss(test_pred, test_y)
    rho_loss = models.PMSModel.density_loss(test_pred, test_y)
    return total_loss, L_loss, A_loss, K_loss, c_loss, rho_loss

def loss_barplot(total_losses, L_losses, A_losses, K_losses, c_losses, rho_losses, title='', labels=None):
    """[summary]

    :param total_losses: [description]
    :type total_losses: [type]
    :param L_losses: [description]
    :type L_losses: [type]
    :param A_losses: [description]
    :type A_losses: [type]
    :param K_losses: [description]
    :type K_losses: [type]
    :param c_losses: [description]
    :type c_losses: [type]
    :param rho_losses: [description]
    :type rho_losses: [type]
    :param title: [description], defaults to ''
    :type title: str, optional
    :param labels: [description], defaults to None
    :type labels: [type], optional
    """    
    x_pos = np.arange(len(L_losses))
    if labels is None:
        labels = x_pos
    fig, ax = plt.subplots(figsize=(15,5), dpi=80)
    width = 0.1
    plot_t = ax.bar(x_pos - 2.5*width, total_losses, width, label='total loss')
    plot_L = ax.bar(x_pos - 1.5*width, L_losses, width, label='L loss')
    plot_A = ax.bar(x_pos - 0.5*width, A_losses, width, label='A loss')
    plot_K = ax.bar(x_pos + 0.5*width, K_losses, width, label='K loss')
    plot_c = ax.bar(x_pos + 1.5*width, c_losses, width, label='c loss')
    plot_r = ax.bar(x_pos + 2.5*width, rho_losses, width, label=r'$\rho$ loss')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_yscale('log')
    ax.set_ylim([1e-6,1e-1])
    ax.set_title(title)
    ax.set_xlabel('output features')
    ax.legend()
    fig.tight_layout()
    plt.show()

def get_colors():
    """[summary]

    :return: [description]
    :rtype: [type]
    """    
    return list(mcolors.TABLEAU_COLORS.values())

def attributions(model, test_x, test_y, prob=False):
    """[summary]

    :param model: [description]
    :type model: [type]
    :param test_x: [description]
    :type test_x: [type]
    :param test_y: [description]
    :type test_y: [type]
    :param prob: [description], defaults to False
    :type prob: bool, optional
    :return: [description]
    :rtype: [type]
    """    
    in_dim, out_dim = (test_x.size(1), 2*test_y.size(1)) if prob else (test_x.size(1), test_y.size(1))
    attr_algo = IntegratedGradients(model)
    baseline = torch.zeros(1, in_dim)
    attr = [f.normalize(torch.abs(attr_algo.attribute(test_x, baseline, target=i)), p=1, dim=1)
                for i in range(out_dim)]
    attrs = torch.stack(attr, dim=1)
    attrs = torch.mean(attrs, dim=0).T
    attrs = attrs[[0,1,2,3,4,5,7],:] # remove rho
    attrs = f.normalize(attrs, p=1, dim=0) # normalize
    return attrs

def plot_attributions(ax, attrs, prob=False):
    """[summary]

    :param ax: [description]
    :type ax: [type]
    :param attrs: [description]
    :type attrs: [type]
    :param prob: [description], defaults to False
    :type prob: bool, optional
    """    
    inputs = [r'$E_1/E_0$', r'$\nu_0$', r'$\nu_1$', r'$\alpha_1/\alpha_0$', r'$\kappa_1/\kappa_0$', r'$c_1/c_0$', r'$f_1$']
    outputs = [r'$\;\underline{\hat{L}}_\mathrm{eff} \rightarrow$', r'$\;\underline{\hat{A}}_\mathrm{eff} \rightarrow$', r'$\;\underline{\hat{K}}_\mathrm{eff} \rightarrow$', r'$c_\mathrm{eff}\;$', r'$\;\rho_\mathrm{eff}$']
    output_sizes = torch.Tensor([21, 6, 6, 1, 1])
    if prob:
        outputs = [r'$L^\sigma$', r'$A^\sigma$', r'$K^\sigma$', r'$c^\sigma$', r'$\rho^\sigma$']
    output_ticks = torch.cat((torch.zeros(1), torch.cumsum(output_sizes, dim=0)))[:-1]
    im = ax.imshow(torch.abs(attrs), cmap='Reds', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(output_ticks)
    ax.set_yticks(torch.arange(len(inputs)))
    ax.set_xticklabels(outputs)
    ax.set_yticklabels(inputs)
    ax.set_xlabel('output features')
    ax.set_ylabel('input features')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im, cax=cax, ticks=[0, 1])

def create_logger(file_path, bash_output=True):
    """[summary]

    :param file_path: [description]
    :type file_path: [type]
    :param bash_output: [description], defaults to True
    :type bash_output: bool, optional
    :return: [description]
    :rtype: [type]
    """    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if bash_output:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger

def get_stats(df, key):
    """[summary]

    :param df: [description]
    :type df: [type]
    :param key: [description]
    :type key: [type]
    :return: [description]
    :rtype: [type]
    """    
    data = [np.array(df_s[key]) for df_s in df]
    d_med = np.array([np.median(d) for d in data])
    d_extrema = [d_med - np.array([np.quantile(d, 0.25) for d in data]), np.array([np.quantile(d, 0.75) for d in data]) - d_med]
    #d_extrema = [d_med - np.array([np.min(d) for d in data]), np.array([np.min(d) for d in data]) - d_med]
    #d_mean = np.mean(data, axis=1)
    #d_std = np.std(data, axis=1)
    return d_med, d_extrema
    #return d_mean, [d_std, d_std]

def plot_model(df, ax, xaxis, prob=False, train_loss=True):
    """[summary]

    :param df: [description]
    :type df: [type]
    :param ax: [description]
    :type ax: [type]
    :param xaxis: [description]
    :type xaxis: [type]
    :param prob: [description], defaults to False
    :type prob: bool, optional
    """    
    name = ''#df['model'].iloc[0]
    df_samples = [y for x, y in df.groupby(xaxis, as_index=False)]
    if prob:
        train_med, train_err = get_stats(df_samples, 'prob_train_loss')
        val_med, val_err = get_stats(df_samples, 'prob_val_loss')
    else:
        train_med, train_err = get_stats(df_samples, 'train_loss')
        val_med, val_err = get_stats(df_samples, 'val_loss')
    train_samples = df[xaxis].unique()
    if train_loss:
        ax.errorbar(train_samples, train_med, yerr=train_err, fmt='--o', label='training loss', capsize=8)
    ax.errorbar(train_samples, val_med, yerr=val_err, fmt='--o', label='validation loss', capsize=8)
    #axs[1,0].set_ylim([1e-7, 1e-4])
    ##axs[1,1].set_ylim([1e-4, 1e-1])
    #ax.set_yscale('log', nonpositive='clip')
    ax.set_xlabel(xaxis)
    if not prob:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend()

def get_archs(df, keys=None, in_dim=8, out_dim=32, prob=False):
    """[summary]

    :param df: [description]
    :type df: [type]
    :param keys: [description], defaults to None
    :type keys: [type], optional
    :param prob: [description], defaults to False
    :type prob: bool, optional
    :return: [description]
    :rtype: [type]
    """    
    loss_key = 'prob_val_loss' if prob else 'val_loss'
    if keys is None:
        keys = ['layers','neurons','activation','train_samples']
    df_archs = [y for x, y in df.groupby(keys, as_index=False)]
    val_loss_med = get_stats(df_archs,'prob_val_loss' if prob else 'val_loss')[0]
    df_archs = df.groupby(keys, as_index=False).first()[keys]
    df_archs[loss_key] = pd.Series(val_loss_med).values
    df_archs['parameters'] = df_archs.neurons * (in_dim + 1) \
        + (df_archs.layers - 1) * df_archs.neurons * (df_archs.neurons + 1) \
        + out_dim * (df_archs.neurons + 1)
    return df_archs

def plot_parameters_loss(ax, df_archs, samples, title='', prob=False):
    """[summary]

    :param df: [description]
    :type df: [type]
    :param samples: [description]
    :type samples: [type]
    :param ax: [description]
    :type ax: [type]
    :param title: [description], defaults to ''
    :type title: str, optional
    :param prob: [description], defaults to False
    :type prob: bool, optional
    """    
    if not isinstance(samples, list):
        samples = [samples]
    for i, train_samples in enumerate(samples):
        loss_key = 'prob_val_loss' if prob else 'val_loss'
        df_filter = df_archs.loc[df_archs['train_samples'] == train_samples]
        ax.axhline(df_filter[loss_key].min(), color=next(ax._get_lines.prop_cycler)['color'])
        sc = ax.scatter(df_filter.parameters, df_filter[loss_key], marker='x', label=f'{train_samples} training samples')
    ax.set_xlabel('number of parameters')
    ax.set_ylabel('median validation loss')
    if not prob:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.legend()
