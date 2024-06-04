import numpy as np
import matplotlib.pyplot as plt
import saqqara
from simulator import LISA_AET
import glob
import torch
import swyft
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from pytorch_lightning import loggers as pl_loggers

DATA_DIR = "./simulations/"
Z_FILES = glob.glob(DATA_DIR + "/z_*.npy")
Z_FILES = sorted(Z_FILES, key=lambda x: str(x.split("_")[-1].split(".")[0]))
DATA_FILES = glob.glob(DATA_DIR + "/cg_data_*.npy")
DATA_FILES = sorted(DATA_FILES, key=lambda x: str(x.split("_")[-1].split(".")[0]))

# Check all names match
for idx in range(len(Z_FILES)):
    assert (
        Z_FILES[idx].split("_")[-1].split(".")[0]
        == DATA_FILES[idx].split("_")[-1].split(".")[0]
    )

# Compute total number of simulations
n_simulations = len(Z_FILES) * 128
print(f"Total number of simulations: {n_simulations}")

# Get data shapes
z_shape = np.load(Z_FILES[0]).shape[1:]
data_shape = np.load(DATA_FILES[0]).shape[1:]
print(f"z shape: {z_shape}")
print(f"data shape: {data_shape}")

config = saqqara.load_settings("default_config.yaml")
sim = LISA_AET(config)

z_dataset = saqqara.NPYDataset(file_paths=Z_FILES)
data_dataset = saqqara.NPYDataset(file_paths=DATA_FILES)
training_dataset = saqqara.TrainingDataset(z_store=z_dataset, data_store=data_dataset)

# Dataset properties
print("Training dataset properties:")
print("length:", len(training_dataset))
print("z shape:", training_dataset["z"][0].shape)
print("data shape:", training_dataset["data"][0].shape)

def setup_dataloaders(
    dataset,
    total_size=None,
    train_fraction=0.8,
    val_fraction=0.2,
    num_workers=0,
    batch_size=64,
):
    if total_size is None:
        total_size = len(dataset)
    indices = list(range(len(dataset)))
    train_idx, val_idx = int(np.floor(train_fraction * total_size)), int(
        np.floor((train_fraction + val_fraction) * total_size)
    )
    train_indices, val_indices = indices[:train_idx], indices[train_idx:val_idx]
    # train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)
    train_sampler, val_sampler = SequentialSampler(train_indices), SequentialSampler(
        val_indices
    )
    train_dataloader = DataLoader(
        dataset=dataset,
        drop_last=True,
        sampler=train_sampler,
        num_workers=int(num_workers),
        batch_size=int(batch_size),
    )
    val_dataloader = DataLoader(
        dataset=dataset,
        drop_last=True,
        sampler=val_sampler,
        num_workers=int(num_workers),
        batch_size=int(batch_size),
    )
    return train_dataloader, val_dataloader

total_size = 100_000
learning_rate = 1e-4
batch_size = 256
train_dl, val_dl = setup_dataloaders(training_dataset, total_size=total_size, num_workers=10, batch_size=batch_size)

# ckpt = "./saqqara/7ibs25vo/checkpoints/epoch=89-step=35100.ckpt"


from swyft.networks import OnlineStandardizingLayer
from swyft.networks import ResidualNetWithChannel
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DNet(nn.Module):
    def __init__(self):
        super(Conv1DNet, self).__init__()
        
        # # First convolutional layer
        # self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # # Second convolutional layer
        # self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # # Third convolutional layer
        # self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # # Fourth convolutional layer
        # self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        # # Fifth convolutional layer
        # self.conv5 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=10, stride=1, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=5, stride=1, padding=1)
        # Third convolutional layer
        self.conv3 = nn.Conv1d(in_channels=12, out_channels=18, kernel_size=5, stride=1, padding=1)
        # Fourth convolutional layer
        self.conv4 = nn.Conv1d(in_channels=18, out_channels=12, kernel_size=2, stride=1, padding=1)
        # Fifth convolutional layer
        self.conv5 = nn.Conv1d(in_channels=12, out_channels=1, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization for each layer
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(12)
        self.bn3 = nn.BatchNorm1d(18)
        self.bn4 = nn.BatchNorm1d(12)
        self.bn5 = nn.BatchNorm1d(1)

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        return x



class InferenceNetwork(swyft.SwyftModule, swyft.AdamWReduceLROnPlateau):
    def __init__(self, sim=None):
        super().__init__()
        self.sim = sim
        self.learning_rate = learning_rate
        self.early_stopping_patience = 100
        self.num_feat_param = 3  # Number of channels
        self.num_params = 4
        self.npts = sim.coarse_grained_f.shape[0]
        self.nl_AA = OnlineStandardizingLayer(shape=(self.npts,))
        self.nl_EE = OnlineStandardizingLayer(shape=(self.npts,))
        self.nl_TT = OnlineStandardizingLayer(shape=(self.npts,))
        self.net = Conv1DNet()
        self.feature_extraction = torch.nn.LazyLinear(2 * self.num_params * self.num_feat_param)
        self.marginals = self.get_marginals(self.num_params)
        self.lrs1d = swyft.LogRatioEstimator_1dim(
            #num_features=self.num_feat_param,
            num_features=2 * self.num_feat_param * self.num_params,
            num_params=self.num_params,
            varnames="z",
            num_blocks=3,
            hidden_features=64,
            dropout=0.1,
        )
        self.lrs2d = swyft.LogRatioEstimator_Ndim(
            #num_features=2 * self.num_feat_param,
            num_features=2 * self.num_feat_param * self.num_params,
            marginals=self.marginals,
            num_blocks=3,
            hidden_features=64,
            varnames="z",
            dropout=0.1,
        )

    def forward(self, A, B):
        log_data = torch.log(A["data"])
        # reshape to (batch, num_channels, num_freqs)
        log_data = log_data.transpose(1, 2)
        norm_AA = self.nl_AA(log_data[..., 0, :])
        norm_EE = self.nl_EE(log_data[..., 1, :])
        norm_TT = self.nl_TT(log_data[..., 2, :])
        full_data = torch.stack([norm_AA, norm_EE, norm_TT], dim=-2)
        s = self.feature_extraction(self.net(full_data)).reshape(-1, 2 * self.num_params * self.num_feat_param)
        lrs1d = self.lrs1d(s, B["z"])
        lrs2d = self.lrs2d(s, B["z"])
        return lrs1d, lrs2d

    @staticmethod
    def get_marginals(n_params):
        marginals = []
        for i in range(n_params):
            for j in range(n_params):
                if j > i:
                    marginals.append((i, j))
        return tuple(marginals)


# class InferenceNetwork(swyft.SwyftModule, swyft.AdamWReduceLROnPlateau):
#     def __init__(self, sim=None):
#         super().__init__()
#         self.sim = sim
#         self.learning_rate = learning_rate
#         self.early_stopping_patience = 100
#         self.num_feat_param = 3  # Number of channels
#         self.num_params = 2 # 4
#         self.npts = sim.coarse_grained_f.shape[0]
#         #self.npts = 650
#         self.nl_AA = OnlineStandardizingLayer(shape=(self.npts,))
#         self.nl_EE = OnlineStandardizingLayer(shape=(self.npts,))
#         self.nl_TT = OnlineStandardizingLayer(shape=(self.npts,))
#         self.nl_AA_nolog = OnlineStandardizingLayer(shape=(self.npts,))
#         self.nl_EE_nolog = OnlineStandardizingLayer(shape=(self.npts,))
#         self.nl_TT_nolog = OnlineStandardizingLayer(shape=(self.npts,))
#         self.resnet = ResidualNetWithChannel(
#             channels=3,
#             in_features=self.npts,
#             out_features=self.num_params,
#             hidden_features=64,
#             num_blocks=2,
#             dropout_probability=0.1,
#             use_batch_norm=True,
#         )
#         self.resnet_no_log = ResidualNetWithChannel(
#             channels=3,
#             in_features=self.npts,
#             out_features=self.num_params,
#             hidden_features=64,
#             num_blocks=2,
#             dropout_probability=0.1,
#             use_batch_norm=True,
#         )
#         self.fc_AA = torch.nn.Linear(self.npts, sim.nparams)
#         self.fc_EE = torch.nn.Linear(self.npts, sim.nparams)
#         self.fc_TT = torch.nn.Linear(self.npts, sim.nparams)
#         self.marginals = self.get_marginals(self.num_params)
#         self.lrs1d = swyft.LogRatioEstimator_1dim(
#             #num_features=self.num_feat_param,
#             num_features=2 * self.num_feat_param * self.num_params,
#             num_params=self.num_params,
#             varnames="z",
#             num_blocks=3,
#             hidden_features=64,
#             dropout=0.1,
#         )
#         self.lrs2d = swyft.LogRatioEstimator_Ndim(
#             #num_features=2 * self.num_feat_param,
#             num_features=2 * self.num_feat_param * self.num_params,
#             marginals=self.marginals,
#             num_blocks=3,
#             hidden_features=64,
#             varnames="z",
#             dropout=0.1,
#         )

#     def forward(self, A, B):
#         log_data = torch.log(A["data"])
#         # reshape to (batch, num_channels, num_freqs)
#         log_data = log_data.transpose(1, 2)
#         norm_AA = self.nl_AA(log_data[..., 0, :])
#         norm_EE = self.nl_EE(log_data[..., 1, :])
#         norm_TT = self.nl_TT(log_data[..., 2, :])

#         full_data = torch.stack([norm_AA, norm_EE, norm_TT], dim=-2)
#         no_log_data = torch.exp(full_data)
#         norm_AA_nolog = self.nl_AA_nolog(no_log_data[..., 0, :])
#         norm_EE_nolog = self.nl_EE_nolog(no_log_data[..., 1, :])
#         norm_TT_nolog = self.nl_TT_nolog(no_log_data[..., 2, :])
#         no_log_data = torch.stack([norm_AA_nolog, norm_EE_nolog, norm_TT_nolog], dim=-2)


#         if False:
#             compression = self.resnet(full_data)
#             s1 = compression.reshape(
#                 -1, self.num_params, self.num_feat_param
#             )  # (batch, num_params, num_feat_param)
#             s2 = torch.stack(
#                 [torch.cat([s1[:, i, :], s1[:, j, :]], dim=-1) for i, j in self.marginals],
#                 dim=1,
#             )
#             lrs1d = self.lrs1d(s1, B["z"])
#             lrs2d = self.lrs2d(s2, B["z"])
#         else:
#             compression = self.resnet(full_data)
#             no_log_compression = self.resnet_no_log(no_log_data)
#             s1 = compression.reshape(-1, self.num_params * self.num_feat_param)
#             s2 = no_log_compression.reshape(-1, self.num_params * self.num_feat_param)
#             s = torch.cat((s1, s2), dim=1)
#             lrs1d = self.lrs1d(s, B["z"][:, :2])
#             lrs2d = self.lrs2d(s, B["z"][:, :2])
#         return lrs1d, lrs2d

#     @staticmethod
#     def get_marginals(n_params):
#         marginals = []
#         for i in range(n_params):
#             for j in range(n_params):
#                 if j > i:
#                     marginals.append((i, j))
#         return tuple(marginals)


logger = pl_loggers.WandbLogger(
    offline=False,
    name=f"conv_net_{total_size}_{learning_rate}_{batch_size}",
    project="saqqara",
    entity="j-b-g-alvey",
    log_model="all",
    config=config,
)
device = "gpu" if torch.cuda.is_available() else "cpu"
trainer = swyft.SwyftTrainer(accelerator=device, max_epochs=100, logger=logger)
network = InferenceNetwork(sim=sim)

def load_network_state(network, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    network.load_state_dict(state_dict)
    return network
ckpt = None
#ckpt = "./saqqara/6lhvf57v/checkpoints/epoch=97-step=38220.ckpt"
#ckpt = "./saqqara/9ifhb4mp/checkpoints/epoch=48-step=9555.ckpt"
#ckpt = "./saqqara/qbh11oi9/checkpoints/epoch=90-step=17745.ckpt"
#ckpt = "./saqqara/jafv7s8y/checkpoints/epoch=96-step=18915.ckpt"
if ckpt is not None:
    network = load_network_state(network, ckpt)
trainer.fit(network, train_dl, val_dl)
