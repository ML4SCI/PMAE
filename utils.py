import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import json
from torch import optim
import torch

# Make 2-dimensional histogram
def make_hist2d(group_num, group_size, step, names, inputs, outputs, scaler, event_type, file_path, lower=None, upper=None):
    inputs = scaler.inverse_transform(inputs)
    outputs = scaler.inverse_transform(outputs)
    if lower is None:
        lower = np.min((outputs[:,group_num*group_size+step], inputs[:,group_num*group_size+step]))
    if upper is None:
        upper = np.max((outputs[:,group_num*group_size+step], inputs[:,group_num*group_size+step]))
    varname = names[group_num*group_size+step]
    heatmap, xedges, yedges = np.histogram2d(inputs[:,group_num*group_size+step],
                                             outputs[:,group_num*group_size+step],
                                             bins=30,
                                             range=[[lower, upper], [lower, upper]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #Plot heatmap
    plt.imshow(heatmap.T,
               extent=extent,
               origin='lower')
    plt.plot([lower, upper],
             [lower, upper],
             color='blue')
    fig = plt.gcf()
    plt.set_cmap('gist_heat_r')
    plt.xlabel('%s True' % varname)
    plt.ylabel('%s Pred' % varname)
    plt.title('Frequency Heatmap (' + event_type + ')')
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.colorbar()
    plt.savefig(file_path + '/hist2d_' + event_type)
    plt.show()

# Custom loss function expects shape [batch_size, num_particles, 4] where 3 items are pt, eta, phi, b-tag
class custom_loss:
    def __init__(self, phi_limit, lower_pt_limit, alpha=0.4, beta=.5, gamma=1., delta=.5, output_vars=3):
        self.phi_limit = phi_limit
        self.lower_pt_limit = lower_pt_limit
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.vars = output_vars
    def compute_loss(self, output, target, zero_padded=[]):
        loss = 0
        for i in range(output.size()[1]):
            if i in zero_padded:
                continue
            elif i % self.vars == 0:
                loss += torch.mean((target[:,i] - output[:,i])**2 + torch.gt(output[:,i], self.lower_pt_limit[i % 4]).long() * \
                    (self.gamma / (1 + torch.exp(-(output[:,i] - self.lower_pt_limit[i % 4]) * 3)) - self.gamma) + \
                        torch.le(output[:,i], self.lower_pt_limit[i % 4]).long()*(self.gamma/2 - self.gamma))
            elif i % self.vars == 1:
                loss += torch.mean((target[:,i] - output[:,i])**2 - output[:,i]**2 * self.beta)
            elif i % self.vars == 2:
                loss += torch.mean(torch.le(torch.abs(output[:,i]), self.phi_limit).long() *\
                    ((torch.sin(((output[:,i] - target[:,i]) / self.phi_limit - .5) * np.pi) + 1)**2 +\
                        (torch.sin(((output[:,i] - target[:,i]) / self.phi_limit - .5) * np.pi) + 1) * 2) * self.alpha +\
                    torch.gt(torch.abs(output[:,i]), self.phi_limit).long() *\
                    (((torch.sin(((self.phi_limit * torch.sign(output[:,i]) - target[:,i]) / self.phi_limit  - .5) * \
                                    np.pi) + 1)**2 +\
                        (torch.sin(((self.phi_limit * torch.sign(output[:,i]) - target[:,i]) / self.phi_limit  - .5) * \
                                np.pi) + 1) * 2) * self.alpha +\
                    (self.phi_limit*torch.sign(output[:,i]) - output[:,i])**2))
            elif i % self.vars == 3:
                if self.vars == 4:
                    loss += torch.mean((target[:,i] - output[:,i])**2) * self.delta
                else:
                    continue
        return loss / (output.size()[1] - len(zero_padded))

# Dataset class
class DataLabelDataset(Dataset):
    def __init__(self, data, labels, dtype: str = 'numpy'):
        super(DataLabelDataset, self).__init__()
        if dtype == 'numpy':
            self.data = torch.from_numpy(data)
            self.labels = torch.from_numpy(labels)
        elif dtype == 'torch':
            self.data = data
            self.labels = labels
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

# Custom SGD optimizer
class SGDWithSaturatingMomentumAndDecay(optim.Optimizer):
    def __init__(self, params, lr=None, momentum=0, max_momentum=0.99, epochs_to_saturate=100, batches_per_epoch=1, weight_decay=0, lr_decay=0.1, min_lr=1e-6, resume_epoch=0):
        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, max_momentum=max_momentum, epochs_to_saturate=epochs_to_saturate, batches_per_epoch=batches_per_epoch, weight_decay=weight_decay, lr_decay=lr_decay, min_lr=min_lr, resume_epoch=resume_epoch)
        super(SGDWithSaturatingMomentumAndDecay, self).__init__(params, defaults)

        for group in self.param_groups:
            # Adjust initial learning rate and momentum based on resume epoch
            steps_to_saturate = group['epochs_to_saturate'] * group['batches_per_epoch']
            resumed_steps = group['resume_epoch'] * group['batches_per_epoch']
            max_momentum = group['max_momentum']
            momentum_step = (max_momentum - group['momentum']) / steps_to_saturate
            group['momentum'] = min(group['momentum'] + momentum_step * resumed_steps, max_momentum)
            group['lr'] = max(group['lr'] * (group['lr_decay'] ** resumed_steps), group['min_lr'])

    def step(self, closure=None):
        for group in self.param_groups:
            steps_to_saturate = group['epochs_to_saturate'] * group['batches_per_epoch']
            max_momentum = group['max_momentum']
            momentum_step = (max_momentum - group['momentum']) / steps_to_saturate

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if group['weight_decay'] != 0:
                    d_p.add_(p.data, alpha=group['weight_decay'])

                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])

            # Increment momentum and decay learning rate after the step
            group['momentum'] = min(group['momentum'] + momentum_step, max_momentum)
            group['lr'] = max(group['lr'] * group['lr_decay'], group['min_lr'])

def parse_model_name(model_name):
    data = {}

    # A dictionary to map from the keys in the model name to the keys in the JSON
    key_map = {
        "DM": "d_model",
        "H": "num_heads",
        "L": "num_layers",
        "F": "d_ff",
        "Dr": "dropout",
        "BS": "batch_size",
        "T": "test_batch_size",
        "RE": "resume_epoch",
        "NE": "num_epochs",
        "ES": "epochs_to_saturate",
        "IM": "init_momentum",
        "MM": "max_momentum",
        "TILR": "tae_init_lr",
        "CILR": "class_init_lr",
        "MSL": "max_seq_len",
        "Mk": "mask",
        "A": "alpha",
        "B": "beta",
        "G": "gamma",
        "D": "delta",
        "OV": "output_vars",
        "WD": "weight_decay",
        "MLR": "min_lr",
        "LD": "lr_decay",
        "CIF": "class_input_features",
        "CFD": "class_ff_dim"
    }

    # Remove 'Model' from the start of the model name
    model_name = model_name.lstrip('Model_')

    # Iterate through each key in the key map
    for key in key_map.keys():
        # If the model name contains the key
        if key in model_name:
            # Find the start and end index of the value
            start = model_name.index(key) + len(key)
            end = model_name.index('_', start) if '_' in model_name[start:] else len(model_name)
            
            # Extract and convert the value
            value = model_name[start:end]
            if 'e' in value or '.' in value:  # The value is a float
                value = float(value)
            else:
                value = int(value)
            
            # Add the key-value pair to the dictionary
            data[key_map[key]] = value
            
            # Remove the processed part from the model name
            model_name = model_name[end+1:]

    return data