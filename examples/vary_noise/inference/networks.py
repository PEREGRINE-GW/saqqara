import torch
import swyft
from swyft.networks import OnlineStandardizingLayer
from swyft.networks import ResidualNetWithChannel
import saqqara


class SignalAET(saqqara.SaqqaraNet):
    def __init__(self, settings={}, sim=None):
        super().__init__(settings=settings)
        if sim is None:
            raise ValueError("Must pass a LISA_AET simulator instance")
        self.sim = sim
        self.num_params = 2  # NOTE: Only training signal parameters
        self.npts = sim.coarse_grained_f.shape[0]
        self.nl_AA = OnlineStandardizingLayer(shape=(self.npts,))
        self.nl_EE = OnlineStandardizingLayer(shape=(self.npts,))
        self.nl_TT = OnlineStandardizingLayer(shape=(self.npts,))
        self.nl_AA_nolog = OnlineStandardizingLayer(shape=(self.npts,))
        self.nl_EE_nolog = OnlineStandardizingLayer(shape=(self.npts,))
        self.nl_TT_nolog = OnlineStandardizingLayer(shape=(self.npts,))
        self.channels = settings["train"].get("channels", "AET")
        self.channel_mask = []
        for c in self.channels:
            if c == "A":
                self.channel_mask.append(0)
            elif c == "E":
                self.channel_mask.append(1)
            elif c == "T":
                self.channel_mask.append(2)
        self.num_feat_param = len(self.channels)  # NOTE: Number of channels (AET)
        self.resnet = ResidualNetWithChannel(
            channels=len(self.channels), # 3,
            in_features=self.npts,
            out_features=self.num_params,
            hidden_features=64,
            num_blocks=2,
            dropout_probability=0.1,
            use_batch_norm=True,
        )
        self.resnet_no_log = ResidualNetWithChannel(
            channels=len(self.channels), # 3,
            in_features=self.npts,
            out_features=self.num_params,
            hidden_features=64,
            num_blocks=2,
            dropout_probability=0.1,
            use_batch_norm=True,
        )
        self.marginals = self.get_marginals(self.num_params)
        # self.target = settings["train"].get("target", None)
        # if self.target is not None and self.target == "1d":
        #     self.lrs2d = swyft.LogRatioEstimator_1dim(
        #         num_features=2 * self.num_feat_param * self.num_params,
        #         num_params=self.num_params,
        #         num_blocks=3,
        #         hidden_features=64,
        #         varnames="z",
        #         dropout=0.1,
        #     )
        self.lrs2d = swyft.LogRatioEstimator_Ndim(
            num_features=2 * self.num_feat_param * self.num_params,
            marginals=self.marginals,
            num_blocks=3,
            hidden_features=64,
            varnames="z",
            dropout=0.1,
        )


    def forward(self, A, B):
        if torch.isnan(A["data"]).any():
            raise ValueError("NaNs in data")
        if torch.isinf(A["data"]).any():
            raise ValueError("Infs in data")
        if (A["data"] < 0).any():
            raise ValueError("Negative values in data")
        log_data = torch.log(A["data"])

        # NOTE: reshape to (batch, num_channels, num_freqs)
        log_data = log_data.transpose(1, 2)
        norm_AA = self.nl_AA(log_data[..., 0, :])
        norm_EE = self.nl_EE(log_data[..., 1, :])
        norm_TT = self.nl_TT(log_data[..., 2, :])
        arr_list = [norm_AA, norm_EE, norm_TT]
        full_data = torch.stack([arr_list[channel] for channel in self.channel_mask], dim=-2)
        
        no_log_data = torch.exp(torch.stack([norm_AA, norm_EE, norm_TT], dim=-2))
        norm_AA_nolog = self.nl_AA_nolog(no_log_data[..., 0, :])
        norm_EE_nolog = self.nl_EE_nolog(no_log_data[..., 1, :])
        norm_TT_nolog = self.nl_TT_nolog(no_log_data[..., 2, :])
        arr_list_no_log = [norm_AA_nolog, norm_EE_nolog, norm_TT_nolog]
        no_log_data = torch.stack([arr_list_no_log[channel] for channel in self.channel_mask], dim=-2)

        compression = self.resnet(full_data)
        no_log_compression = self.resnet_no_log(no_log_data)
        s1 = compression.reshape(-1, self.num_params * self.num_feat_param)
        s2 = no_log_compression.reshape(-1, self.num_params * self.num_feat_param)
        s = torch.cat((s1, s2), dim=1)
        lrs2d = self.lrs2d(s, B["z"][:, :2])  # NOTE: Only need 2d logratio
        return lrs2d  # NOTE: Training 2d logratio only here

    @staticmethod
    def get_marginals(n_params):
        marginals = []
        for i in range(n_params):
            for j in range(n_params):
                if j > i:
                    marginals.append((i, j))
        return tuple(marginals)
