####################################################################################################

## IMPORTS ##

# Numpy first! (required to force an Intel OpenMP threading layer)
import numpy as np

# Matplotlib second, to be compatible with headless setups
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Basics
import os

# Scientific computing
import numpy as np
import scipy as sp

# Data handling
import vaex as vx

# Plotting
import matplotlib

# Neural Networks / Neural ODEs
import torch
import torchdiffeq as thdeq
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F
import torch.distributions as distributions
import torch.utils.data.dataloader as DataLoader
from torch import optim
import torch.cuda.amp as amp
from torch._six import inf
from torchsummary import summary

# Self-rolled utility functions
from src.util.datamanip import data_by_tick, data_by_tick_col
from src.util.plotting import data_timeplot
from src.util.datasets import StockDataset
from pyromaniac.optim.torch.adamwcd import AdamWCD as AdamWCD
from torch import optim

####################################################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(" ")
print("Device: {}".format(device))
print(" ")
print(" ")

####################################################################################################

## HYPERPARAMETERS ##
batch_size = 1
input_datasize = 1
hidden_size_gruode = 50 * (input_datasize + 1)
hidden_size_ffw_f = 275
hidden_size_ffw_g = 75
hidden_size_ffw_onn = 75
output_datasize = input_datasize
ttsr = 0.9994
window_size = 9350
max_train_len = 25

####################################################################################################

## MODULES / CLASSES ##


class f(torch.nn.Module):
    def __init__(self, size, hidden_size):
        super(f, self).__init__()
        self.fc1 = nn.Linear(size + 1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, size)

    def forward(self, t, data):
        x = torch.cat(
            (torch.tensor([t], requires_grad=True).to(device), data.to(device)), dim=0
        ).to(device)
        x.retain_grad()
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x


################################################################################


class CustomOdeint(torch.nn.Module):
    def __init__(self, size, hidden_size):
        super(CustomOdeint, self).__init__()
        self.size = size
        self.hidden_size = hidden_size
        self.f_forward = f(self.size, self.hidden_size).to(device)

    def forward(self, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None):
        b_size = len(y0)
        assert b_size == len(t)
        output = torch.Tensor(b_size, t.size()[1], self.size).to(device)
        for i in range(0, b_size):
            output[i] = odeint(self.f_forward, y0[i], t[i], rtol, atol, method, options)
        return output


################################################################################

customodeint = CustomOdeint(hidden_size_gruode, hidden_size_ffw_f)

################################################################################


class GRUODE(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, bias=True
    ):  # hidden_size == output_size of Sampler class
        super(GRUODE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.GRU = torch.nn.GRUCell(self.input_size, self.hidden_size, self.bias).to(
            device
        )
        #self.LN = torch.nn.LayerNorm(hidden_size)

    def forward(self, x, hidden, times):
        # GRU training algorithm
        h_n = customodeint(
            hidden.to(device), times.to(device), rtol=1e-3, atol=1e-4, method="dopri5"
        )[
            :, 1
        ]  # check
        #hy = self.LN(h_n)
        hy = self.GRU(x, h_n)
        return hy


################################################################################


class g(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size=2
    ):  # input_size == hidden_size of GRUODE
        super(g, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        #self.BN = nn.BatchNorm1d(hidden_size)
        self.output_size = output_size

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, (1.0 / 5.5))
        x = self.fc2(x)
        x = F.leaky_relu(x, (1.0 / 5.5))
        #x = self.BN(x)
        x = self.fc3(x)
        x = (
            torch.cat((x[:, 0], torch.abs(x[:, 1])), dim=0)
            .to(device)
            .view(-1, self.output_size)
        )
        return x


################################################################################


class Sampler(torch.nn.Module):
    def __init__(
        self, dist, nr_parameters, output_size
    ):  # dim nr_parameters != dim output_size
        super(Sampler, self).__init__()
        self.dist = dist
        self.nr_parameters = nr_parameters
        self.output_size = output_size

    def forward(self, parameters):
        parameters = parameters.t()
        parlist = parameters.chunk(self.nr_parameters)
        distribution = self.dist(*parlist)  # unpacking elements of parlist
        out = distribution.rsample(torch.tensor([self.output_size]))[
            :, 0
        ]  # non vuole requires_grad()
        out = out.t()
        return out


################################################################################


class OutputNN(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size
    ):  # dim input_size == hidden_size of GRUODE, dim output_size == dim predictions
        super(OutputNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, (1.0 / 5.5))
        x = self.fc2(x)
        x = F.leaky_relu(x, (1.0 / 5.5))
        x = self.fc3(x)
        return x


################################################################################


class RNNVAEODE(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size_gruode,
        hidden_size_ffw_g,
        hidden_size_ffw_onn,
        prior,
        h0,
    ):
        super(RNNVAEODE, self).__init__()

        self.h0 = h0
        self.h0_buffer = h0
        self.GRUODE = GRUODE(input_size, hidden_size_gruode, bias=True)
        self.g = g(hidden_size_gruode, hidden_size_ffw_g, output_size=2)
        self.Sampler = Sampler(prior, 2, hidden_size_gruode)
        self.OutputNN = OutputNN(hidden_size_gruode, hidden_size_ffw_onn, input_size)

    def set_h0(self, new_h0):
        self.h0 = new_h0.detach().clone()
        self.h0_buffer = new_h0.detach().clone()

    def forward(self, past, t_future):

        self.h0 = self.h0_buffer.detach().clone()

        if past is not None:  # training
            past_time = past[:, :, :1]
            past_data = past[:, :, 1:]

            h_new = self.GRUODE(
                past_data[:, 0],
                self.h0,
                torch.cat(
                    (
                        past_time[:, 0] - torch.ones(past_time[:, 0].size()).to(device),
                        past_time[:, 0],
                    ),
                    dim=1,
                ).to(device),
            )
            h_prev = h_new

            for i in range(1, len(past)):
                h_new = self.GRUODE(
                    past_data[:, i],
                    h_prev,
                    torch.cat((past_time[:, i - 1], past_time[:, i]), dim=1).to(device),
                )
                h_prev = h_new

            self.h0_buffer = h_prev.detach().clone()

        else:
            h_prev = self.h0  # not rolled

        if t_future is None:
            return None  # not rolled

        param = self.g(h_prev)
        z0 = self.Sampler(param)
        out = customodeint(z0, t_future, rtol=1e-4, atol=1e-5, method="dopri5")[:, :]
        output = self.OutputNN(out)
        return output


####################################################################################################

## INITIAL HIDDEN STATES ##

h0 = torch.zeros(batch_size, hidden_size_gruode)
h0.to(device)

#h0_test = torch.zeros(1, hidden_size_gruode)
#h0_test.to(device)

####################################################################################################

## DATASETS ##

mydataset = StockDataset(
    "./data/WIKI_PRICES_QUANDL.hdf5",
    "AAPL",
    "close",
    ttsr,
    window_size,
    batch_size=batch_size,
)

mydataset_test = StockDataset(
    "./data/WIKI_PRICES_QUANDL.hdf5", "AAPL", "close", ttsr, window_size, train=False
)

####################################################################################################

## MODEL INSTANTIATION ##

model = RNNVAEODE(
    input_datasize,
    hidden_size_gruode,
    hidden_size_ffw_g,
    hidden_size_ffw_onn,
    distributions.Normal,
    h0,
)
model.to(device)

####################################################################################################

## DATALOADERS ##

dataloader = torch.utils.data.DataLoader(mydataset, batch_size=batch_size)
dataloader_test = torch.utils.data.DataLoader(mydataset_test, batch_size=1)

####################################################################################################

## UTILITIES ##


def param_gn(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


################################################################################


def get_params_num(net):
    return sum(map(torch.numel, net.parameters()))


####################################################################################################

## GREETS ##

print(" ")
print("Total parameter nr.:", get_params_num(model))
print(" ")
print(" ")

####################################################################################################

# TRAINING LOOP
lr = 0.01
eps = 1e-8
cn = 10
out_cn = 4 * cn
crazy_cn = 2 * out_cn
# wd = 0.00325
wd = 0.00163
gammadec = 0.50
gammastep = 2
epochs = max_train_len

printout_step = 1

n_batches = len(dataloader)

# criterion = nn.SmoothL1Loss(reduction="mean")
# optimizer = optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=wd)
criterion = nn.L1Loss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=gammastep, gamma=gammadec)

model.train()

clip_counter = 1

for e in range(epochs):
    for i, dictionary in enumerate(dataloader):
        clip_iter = 3
        optimizer.zero_grad()
        data_in = dictionary["past"].float().to(device)
        data_out_time = dictionary["future"][:, :, :1].float().to(device)
        data_out_data = dictionary["future"][:, :, 1:].float().to(device)

        outputs = model(data_in, data_out_time)
        loss = criterion(outputs, data_out_data)
        loss.backward()
        paramgn = param_gn(model.parameters())

        # PRINTOUT
        if i % printout_step == 0:
            print(" ")
            print("[UNCLIPPED GRAD NORM]: ", paramgn.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), cn)

        # ADAPTIVE GRADIENT CLIPPING
        """
        if paramgn > crazy_cn:
            torch.nn.utils.clip_grad_norm_(model.parameters(), out_cn)
            clip_counter = 5
        elif paramgn > out_cn:
            clip_iter = 2
            if clip_counter % clip_iter == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cn)
                clip_counter = 8
        else:
            clip_iter = 3
            if clip_counter % clip_iter == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cn)
                clip_counter = 0

        clip_counter += 1
        """

        # PRINTOUT
        if i % printout_step == 0:
            paramgn = param_gn(model.parameters())
            print("[CLIPPED GRAD NORM]: ", paramgn.item())

        optimizer.step()
        scheduler.step()

        # PRINTOUT
        if i % printout_step == 0:
            print(
                "[EPOCH]: {}, [BATCH]: {}/{}, [LOSS]: {}".format(
                    e, i, n_batches, loss.item()
                )
            )

        print(" ")
        print(" ")

# END OF THE LOOP
# Save the model
torch.save(model.state_dict(), "RNNVAEODE_bkp_carapax_AAPLCLOSE.pt")

####################################################################################################

## TESTING LOOP ##

#model_test = model
#model_test.eval()
#model_test.set_h0(h0[1])
#model.set_h0(h0[1])
criterion = nn.L1Loss()
for i, dictionary in enumerate(dataloader_test):
    if dictionary["future"] == []:
        #data_in = dictionary["past"].float().to(device)
        #data_out_time = None
        #outputs = model_test(data_in, data_out_time)
        pass
    else:
        data_in = None
        data_out_time = dictionary["future"][:, -5:, :1].float().to(device)
        data_out_data = dictionary["future"][:, -5:, 1:].float().to(device)
        #outputs = model_test(data_in, data_out_time)
        outputs = model(data_in, data_out_time)
        loss = criterion(outputs, data_out_data)
        #print("[FINAL TEST LOSS]: ", loss)
        #print("True: ", data_out_data)
        #print("Predicted: ", outputs)

####################################################################################################

model_test = model
model_test.eval()
lista = list()
criterion = nn.L1Loss()

uncertainty_fan = 100

for j in range(uncertainty_fan):
  for i,dictionary in enumerate(dataloader_test):
    if dictionary["future"] == []:
      data_in = dictionary["past"].float().to(device)
      data_out_time = None
      outputs = model_test(data_in,data_out_time)
      lista.append(outputs.reshape(1,-1)[0])
    else:
      data_in = None
      data_out_time = dictionary["future"][:,:,:1].float().to(device)
      data_out_data = dictionary["future"][:,:,1:].float().to(device)
      outputs = model_test(data_in,data_out_time)
      loss = criterion(outputs, data_out_data)
      lista.append(outputs.reshape(1,-1)[0].detach().cpu().numpy() )

  data_out_time  = torch.tensor([mydataset_test[len(mydataset_test)-1]["future"][:,:1]])
  out = model_test(None,data_out_time)
  plt.plot(data_out_time[0].detach().cpu().numpy(), out[0].detach().cpu().numpy())
l = np.array(lista)
l = l[l[:,0].argsort(), :]  # sorts arrays inside the array of arrays l, based on inner array's first value (index 0)

l_ = l.copy()
l_.sort(0)                  # Sorts accordind to desc order, element by element of the columns

if (uncertainty_fan%2):
    index = int(floor(uncertainty_fan/2.0))
    traj = l[index]
    traj_ = l_[index]
else:
    index = int(uncertainty_fan/2.0)
    traj = (l[index] + l[index+1])/2.0
    traj_ = (l_[index] + l_[index+1])/2.0


loss_final_mic = criterion(outputs, torch.tensor(traj).reshape(outputs.size()).to(device))     # MIC == Median on Initial Condition
loss_final_mec = criterion(outputs, torch.tensor(traj_).reshape(outputs.size()).to(device))    # MEC == Median on Every Condition

print("[MEDIAN I.C. TEST LOSS]: ", loss_final_mic)
print("[MEDIAN E.C. TEST LOSS]: ", loss_final_mec)
data_out_data  = torch.tensor([mydataset_test[len(mydataset_test)-1]["future"][:,1:]])
plt.plot(data_out_time[0].detach().cpu().numpy(), data_out_data[0].detach().cpu().numpy(), color='black', marker='o', linestyle='dashed', linewidth=2, markersize=12, label="True data")
plt.plot(data_out_time[0].detach().cpu().numpy(), traj, color='red', marker='x', linestyle='dashed', linewidth=2, markersize=12, label="Best prediction (median I.C.)")
plt.plot(data_out_time[0].detach().cpu().numpy(), traj_, color='blue', marker='*', linestyle='dashed', linewidth=2, markersize=12, label="Best prediction (median E.C.)")
plt.legend()
plt.title(label="AAPL close price prediction")
plt.xlabel(xlabel="Days")
plt.ylabel(ylabel="Normalized close price")

####################################################################################################

plt.savefig('AAPLOPEN.png')

