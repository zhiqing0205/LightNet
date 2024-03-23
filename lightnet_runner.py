import logging
from collections import OrderedDict

import torch
from torch import optim
from tqdm import tqdm
import os
import numpy as np
import metrics

from dataset import get_data_loader
from models import feature_extractor, feature_fusion, noise_generator, discriminator
from utils.mvtec3d_util import organized_pc_to_unorganized_pc
import math
from models.pointnet2_utils import interpolating_points
import common
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


class TBWrapper:

    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                                s + 2 * padding - 1 * (self.patchsize - 1) - 1
                        ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x


class LightNet:
    def __init__(self, args):
        self.args = args
        self.image_size = args.img_size
        self.count = args.max_sample
        self.device = args.device
        self.auto_noise = args.auto_noise
        self.dsc_lr = args.dsc_lr
        self.gan_epochs = args.gan_epochs
        self.mix_noise = args.mix_noise
        self.noise_std = args.noise_std
        self.meta_epochs = args.meta_epochs
        # print('meta_epochs: ', self.meta_epochs)
        # AED
        self.aed_meta_epochs = args.aed_meta_epochs
        self.dsc_margin = args.dsc_margin
        self.patch_maker = PatchMaker(3, stride=1)

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=(self.image_size, self.image_size)
        )

        # 特征提取器
        self.feature_extractor = feature_extractor.FeatureExtractorBlock(device=self.device,
                                                                         rgb_backbone_name=self.args.rgb_backbone_name,
                                                                         xyz_backbone_name=self.args.xyz_backbone_name,
                                                                         group_size=args.group_size,
                                                                         num_group=args.num_group).to(self.device)
        # 始终为eval模式
        self.feature_extractor.eval()

        # 特征融合
        self.feature_fusion = feature_fusion.FeatureFusionBlock(1152, 768, mlp_ratio=4.).to(self.device)

        # 读取ckpt
        self.feature_fusion.load_state_dict(torch.load('checkpoints/uff_pretrain.pth', map_location=self.device))

        # 始终为eval模式
        self.feature_fusion.eval()

        # 噪声生成器
        self.noise_generator = noise_generator.NoiseGenerator(1152 + 768, 2, hidden=2048,
                                                              mix_noise=self.mix_noise, noise_std=self.noise_std).to(
            self.device)

        # parameters = list(self.feature_extractor.parameters()) + \
        #              list(self.feature_fusion.parameters()) + \
        #              list(self.noise_generator.parameters())

        parameters = self.noise_generator.parameters()

        # 定义优化器
        # self.gen_opt = optim.Adam(parameters, lr=0.001)
        self.gen_opt = optim.AdamW(parameters, lr=0.003, betas=(0.9, 0.95))
        # torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, betas=(0.9, 0.95))

        # 判别器
        self.discriminator = discriminator.Discriminator((1152 + 768), n_layers=2, hidden=1024).to(self.device)
        self.auto_noise = [self.auto_noise, None]
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt,
                                                                   (
                                                                           self.meta_epochs - self.aed_meta_epochs) * self.gan_epochs,
                                                                   self.dsc_lr * .4)
        self.cos_lr = True

        # 日志
        self.logger = TBWrapper('./logs')
        # 模型保存路径
        self.ckpt_dir = 'model_save'
        # 特征保存路径
        self.feature_dir = 'feature'

        self.average = torch.nn.AvgPool2d(3, stride=1)  # torch.nn.AvgPool2d(1, stride=1) #
        self.resize = torch.nn.AdaptiveAvgPool2d((56, 56))
        self.resize2 = torch.nn.AdaptiveAvgPool2d((56, 56))

        # self.ins_id = 0

    def fit(self, class_name):
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size, args=self.args)
        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size, args=self.args)

        # print('fit..')
        # print(len(train_loader))
        self.train(train_loader, test_loader, class_name)
        # for sample, _ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
        #     # fusion_patch, fake_feats, contrastive_loss = self._predict(sample, class_name)
        #     # print('Fusion patch: ', fusion_patch)
        #     # print('fake_feats: ', fake_feats)
        #     # print('contrastive_loss: ', contrastive_loss)
        #     print(len(sample))
        #     print(len(sample[0]))
        #     self.train(sample, sample)
        #     exit()

    # 生成器前向推理
    def _predict(self, sample, class_name=None, is_test=True):
        # print('in _predict')
        # print(len(sample))
        # print(sample)
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]

        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)

        # 特征提取
        with torch.no_grad():
            rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx = self.feature_extractor(
                sample[0],
                unorganized_pc_no_zeros.contiguous())

        xyz = unorganized_pc_no_zeros.to(self.device)
        interpolated_pc = interpolating_points(xyz.contiguous(), center.permute(0, 2, 1), xyz_feature_maps).to("cpu")

        xyz_feature_maps = [fmap.to("cpu") for fmap in [xyz_feature_maps]]
        rgb_feature_maps = [fmap.to("cpu") for fmap in [rgb_feature_maps]]

        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        # print('rgb_patch.shape: ', rgb_patch.shape)
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T

        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        xyz_patch2 = xyz_patch2.to(self.device)
        rgb_patch2 = rgb_patch2.to(self.device)

        # print(f'xyz_patch2.shape: ', xyz_patch2.shape)
        # print(f'rgb_patch2.shape: ', rgb_patch2.shape)

        # print('*' * 60)
        # 特征融合
        # if is_test:
        if True:
            with torch.no_grad():
                fusion_patch = self.feature_fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
                # contrastive_loss = self.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
        else:
            fusion_patch = self.feature_fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            contrastive_loss = self.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))

        fusion_patch = fusion_patch.reshape(-1, fusion_patch.shape[2]).detach()
        if is_test:
          with torch.no_grad():
            fake_feats = self.noise_generator(fusion_patch)
        else:
          fake_feats = self.noise_generator(fusion_patch)
        # print('*' * 60)
        # print(fusion_patch, fake_feats, contrastive_loss)
        # print('Fusion patch: ', fusion_patch)
        # print('fake_feats: ', fake_feats)
        # print('contrastive_loss: ', contrastive_loss)
        # print(f'fusion_patch.shape: ', fusion_patch.shape)


        # return fusion_patch, fake_feats, contrastive_loss
        return fusion_patch, fake_feats, torch.tensor(0)

    def evaluate(self, class_name):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size, args=self.args)
        path_list = []
        with torch.no_grad():
            for sample, mask, label, rgb_path in tqdm(test_loader,
                                                      desc=f'Extracting test features for class {class_name}'):
                pass
        return image_rocaucs, pixel_rocaucs, au_pros

    def train(self, training_data, test_data, class_name):

        state_dict = {}
        ckpt_path = os.path.join(self.ckpt_dir, f"{class_name}_ckpt.pth")

        # if os.path.exists(ckpt_path):
        #     state_dict = torch.load(ckpt_path, map_location=self.device)
        #     if 'discriminator' in state_dict:
        #         self.discriminator.load_state_dict(state_dict['discriminator'])
        #         if "pre_projection" in state_dict:
        #             self.pre_projection.load_state_dict(state_dict["pre_projection"])
        #     else:
        #         self.load_state_dict(state_dict, strict=False)
        #
        #     self.predict(training_data, "train_")
        #     scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
        #     auroc, full_pixel_auroc, anomaly_pixel_auroc = self._evaluate(test_data, scores, segmentations, features,
        #                                                                   labels_gt, masks_gt)
        #
        #     return auroc, full_pixel_auroc, anomaly_pixel_auroc

        def update_state_dict(d):

            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            state_dict["feature_fusion"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.feature_fusion.state_dict().items()})
            state_dict["noise_generator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.noise_generator.state_dict().items()})

        best_record = None
        # print('meta_epochs: ', self.meta_epochs)
        for i_epoch in range(self.meta_epochs):

            self._train_discriminator(training_data)

            self._train_generator(training_data)

            # torch.cuda.empty_cache()
            scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)
            auroc, full_pixel_auroc, pro = self._evaluate(test_data, scores, segmentations, features, labels_gt,
                                                          masks_gt)
            self.logger.logger.add_scalar("i-auroc", auroc, i_epoch)
            self.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_epoch)
            self.logger.logger.add_scalar("pro", pro, i_epoch)

            if best_record is None:
                best_record = [auroc, full_pixel_auroc, pro]
                update_state_dict(state_dict)
                # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
            else:
                if auroc > best_record[0]:
                    best_record = [auroc, full_pixel_auroc, pro]
                    update_state_dict(state_dict)
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
                elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                    best_record[1] = full_pixel_auroc
                    best_record[2] = pro
                    update_state_dict(state_dict)
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})

            print(f"----- {i_epoch + 1} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                  f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                  f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----")

        torch.save(state_dict, ckpt_path)

        return best_record

    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""

        self.discriminator.train()
        # self.feature_fusion.eval()
        self.noise_generator.eval()

        i_iter = 0
        LOGGER.info(f"Training discriminator...")
        with tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                for data_item in input_data:
                    # print(len(data_item))
                    # print(data_item)
                    self.dsc_opt.zero_grad()
                    # self.gen_opt.zero_grad()

                    i_iter += 1
                    # img = data_item["image"]
                    # img = img.to(torch.float).to(self.device)
                    true_feats, fake_feats, _ = self._predict(data_item[0])

                    # print(f'true_feats: {true_feats.shape}, fake_feats: {fake_feats.shape}')
                    # print('torch.cat([true_feats, fake_feats]: ', len(torch.cat([true_feats, fake_feats])))
                    scores = self.discriminator(torch.cat([true_feats, fake_feats]))
                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(fake_feats):]

                    th = self.dsc_margin
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                    true_loss = torch.clip(-true_scores + th, min=0)
                    fake_loss = torch.clip(fake_scores + th, min=0)

                    self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                    self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)

                    loss = true_loss.mean() + fake_loss.mean()
                    self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    self.logger.step()

                    loss.backward()
                    # self.gen_opt.step()
                    self.dsc_opt.step()

                    loss = loss.detach().cpu()
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())

                if self.cos_lr:
                    self.dsc_schl.step()

                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)

    def _train_generator(self, input_data):
        self.discriminator.eval()
        # self.feature_fusion.train()
        self.noise_generator.train()

        i_iter = 0
        LOGGER.info(f"Training generator...")
        all_loss = []
        all_p_true = []
        all_p_fake = []
        all_contrastive_loss = []
        for data_item in input_data:
            # print(len(data_item))
            # print(data_item)
            self.gen_opt.zero_grad()
            # self.gen_opt.zero_grad()

            i_iter += 1
            # img = data_item["image"]
            # img = img.to(torch.float).to(self.device)
            true_feats, fake_feats, contrastive_loss = self._predict(data_item[0], is_test=False)

            # print(f'true_feats: {true_feats.shape}, fake_feats: {fake_feats.shape}')
            scores = self.discriminator(torch.cat([true_feats, fake_feats]))
            true_scores = scores[:len(true_feats)]
            fake_scores = scores[len(fake_feats):]

            th = self.dsc_margin
            p_true = (true_scores.detach() >= th).sum() / len(true_scores)
            p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
            true_loss = torch.clip(-true_scores + th, min=0)
            fake_loss = torch.clip(fake_scores + th, min=0)

            self.logger.logger.add_scalar(f"gen_p_true", p_true, self.logger.g_iter)
            self.logger.logger.add_scalar(f"gen_p_fake", p_fake, self.logger.g_iter)
            self.logger.logger.add_scalar(f"contrastive_loss", contrastive_loss, self.logger.g_iter)

            loss = contrastive_loss - true_loss.mean() - fake_loss.mean()
            self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
            self.logger.step()

            loss.backward()
            self.gen_opt.step()

            loss = loss.detach().cpu()
            all_loss.append(loss.item())
            all_p_true.append(p_true.cpu().item())
            all_p_fake.append(p_fake.cpu().item())
            all_contrastive_loss.append(contrastive_loss.item())

        all_loss = sum(all_loss) / len(input_data)
        all_p_true = sum(all_p_true) / len(input_data)
        all_p_fake = sum(all_p_fake) / len(input_data)
        all_contrastive_loss = sum(all_contrastive_loss) / len(input_data)
        cur_lr = self.gen_opt.state_dict()['param_groups'][0]['lr']
        pbar_str = f"loss:{round(all_loss, 5)} "
        pbar_str += f"lr:{round(cur_lr, 6)}"
        pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
        pbar_str += f"contrastive_loss:{round(all_contrastive_loss, 5)}"

    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):

        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
        # scores = np.mean(scores, axis=0)

        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )["auroc"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            norm_segmentations = np.zeros_like(segmentations)
            for min_score, max_score in zip(min_scores, max_scores):
                norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
            norm_segmentations = norm_segmentations / len(scores)

            # Compute PRO score & PW Auroc for all images
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                norm_segmentations, masks_gt)
            # segmentations, masks_gt
            full_pixel_auroc = pixel_scores["auroc"]

            pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)),
                                      norm_segmentations)
        else:
            full_pixel_auroc = -1
            pro = -1

        return auroc, full_pixel_auroc, pro

    def predict(self, test_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.discriminator.eval()
        # self.feature_fusion.eval()
        self.noise_generator.eval()

        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []

        # with tqdm(test_dataloader, desc="Inferring...", leave=False) as data_iterator:
        #     for data in data_iterator:
        #         if isinstance(data, dict):
        #             labels_gt.extend(data["is_anomaly"].numpy().tolist())
        #             if data.get("mask", None) is not None:
        #                 masks_gt.extend(data["mask"].numpy().tolist())
        #             image = data["image"]
        #             img_paths.extend(data['image_path'])
        #         _scores, _masks, _feats = self._predict(image)
        #         for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, data["is_anomaly"].numpy().tolist()):
        #             scores.append(score)
        #             masks.append(mask)]

        with tqdm(test_dataloader, desc="Inferring...", leave=False) as data_iterator:
            for sample, mask_gt, label, rgb_path in data_iterator:
                # print(f'sample: {sample}')
                # print(f'sample[0] - img', {sample[0].shape})
                # print(f'sample[1] - resized_organized_pc', {sample[1].shape})
                # print(f'sample[2] - resized_depth_map_3channel: {sample[2].shape}')
                # print(f'mask_gt.shape: {mask_gt.shape}')
                # print(f'label.shape: {label.shape}')
                # print(f'rgb_path: {rgb_path}')
                # exit()
                img_paths.extend(rgb_path)
                labels_gt.extend(label.numpy().tolist())
                if mask_gt is not None:
                    masks_gt.extend(mask_gt.numpy().tolist())
                fusion_patch, _, _ = self._predict(sample)
                _scores, _masks, _feats = self.dsc_predict(fusion_patch)
                for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, label.numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)
                    # features.append(feat)

        return scores, masks, features, labels_gt, masks_gt

    def dsc_predict(self, fusion_patch):
        """Infer score and mask for a batch of images."""
        fusion_patch = fusion_patch.to(torch.float).to(self.device)

        # 为fusion_patch添加第一维为1
        # fusion_patch = fusion_patch.reshape(1, fusion_patch.shape[0], fusion_patch.shape[1])
        # batchsize = fusion_patch.shape[0]
        batchsize = 1
        self.discriminator.eval()
        with torch.no_grad():
            # features, patch_shapes = self._embed(images,
            #                                      provide_patch_shapes=True,
            #                                      evaluation=True)
            features, patch_shapes = fusion_patch, [[56, 56]]

            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            patch_scores = image_scores = -self.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            # print(f'batchsize: {batchsize}, scales: {scales}')
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        # print('*'*60)
        # print(list(image_scores), list(masks), list(features))

        return list(image_scores), list(masks), list(features)
