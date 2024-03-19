import torch

from models.utils import init_weight


class NoiseGenerator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None, mix_noise=1, noise_std=0.015):
        super(NoiseGenerator, self).__init__()
        self.mix_noise = mix_noise
        self.noise_std = noise_std

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        # print('in_planes', in_planes)
        self.tail = torch.nn.Linear(_hidden, in_planes, bias=False)
        self.apply(init_weight)

    def forward(self, true_feats):
        noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to('cuda')  # (N, K)
        noise = torch.stack([
            torch.normal(0, self.noise_std * 1.1 ** k, true_feats.shape)
            for k in range(self.mix_noise)], dim=1).to('cuda')  # (N, K, C)
        # print('noise.shape', noise.shape)
        # print('noise_one_hot', noise_one_hot.shape)
        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
        # fake_feats = true_feats + noise
        # print('noise.shape', noise.shape)
        # # 将噪声拼接到true_feats上
        # fake_feats = torch.cat([true_feats, noise], dim=-1)
        # 为了与原始特征兼容，我们不再拼接噪声，而是直接加到true_feats上
        fake_feats = true_feats + noise
        # print('fake_feats.shape', fake_feats.shape)
        fake_feats_new = self.tail(self.body(fake_feats))
        # print(f'fake_feats_new.shape', fake_feats_new.shape)
        return fake_feats_new
