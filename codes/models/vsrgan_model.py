from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.optim as optim
from math import exp
from .vsr_model import VSRModel
from .networks import define_generator, define_discriminator
from .networks.vgg_nets import VGGFeatureExtractor
from .optim import define_criterion, define_lr_schedule
from utils import net_utils


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def loss_mse_ssim_crit(x, y):
    x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
    ssim_loss = 1e-1 * (1 - torch.mean(ssim(x, y)))
    # mae_loss = 0.2 * torch.mean(torch.abs(x - y))
    # mse_loss = 1 * torch.mean(torch.square(y - x))
    loss = ssim_loss
    return loss

class VSRGANModel(VSRModel):
    """ A model wraper for subjective video super-resolution

        It contains a generator and a discriminator, as well as relative
        functions to train and test the generator
    """

    def __init__(self, opt):
        super(VSRGANModel, self).__init__(opt)

        if self.is_train:
            self.cnt_upd_D = 0

    def set_network(self):
        # define net G
        self.net_G = define_generator(self.opt).to(self.device)
        if self.verbose:
            self.logger.info('Generator: {}\n'.format(
                self.opt['model']['generator']['name']) + self.net_G.__str__())

        # load net G
        load_path_G = self.opt['model']['generator'].get('load_path')
        if load_path_G is not None:
            self.load_network(self.net_G, load_path_G)
            if self.verbose:
                self.logger.info('Loaded generator from: {}'.format(load_path_G))

        if self.is_train:
            # define net D
            self.net_D = define_discriminator(self.opt).to(self.device)
            if self.verbose:
                self.logger.info('Discriminator: {}\n'.format(
                    self.opt['model']['discriminator']['name']) + self.net_D.__str__())

            # load net D
            load_path_D = self.opt['model']['discriminator'].get('load_path')
            if load_path_D is not None:
                self.load_network(self.net_D, load_path_D)
                if self.verbose:
                    self.logger.info('Loaded discriminator from: {}'.format(
                        load_path_D))

    def config_training(self):
        # set criterions
        self.set_criterion()

        # set optimizer for G
        lr = self.opt['train']['generator']['lr']
        weight_decay = self.opt['train']['generator'].get('weight_decay', 0)
        betas = (
            self.opt['train']['generator'].get('beta1', 0.9),
            self.opt['train']['generator'].get('beta2', 0.999))
        self.optim_G = optim.Adam(
            self.net_G.parameters(),
            lr=lr, weight_decay=weight_decay, betas=betas)

        # set optimizer for D
        lr = self.opt['train']['discriminator']['lr']
        weight_decay = self.opt['train']['discriminator'].get('weight_decay', 0)
        betas = (
            self.opt['train']['discriminator'].get('beta1', 0.9),
            self.opt['train']['discriminator'].get('beta2', 0.999))
        self.optim_D = optim.Adam(
            self.net_D.parameters(),
            lr=lr, weight_decay=weight_decay, betas=betas)

        # set lr schedules for G
        lr_schedule = self.opt['train']['generator'].get('lr_schedule')
        self.sched_G = define_lr_schedule(lr_schedule, self.optim_G)

        # set lr schedules for D
        lr_schedule = self.opt['train']['discriminator'].get('lr_schedule')
        self.sched_D = define_lr_schedule(lr_schedule, self.optim_D)

    def set_criterion(self):
        # pixel criterion
        self.pix_crit = define_criterion(
            self.opt['train'].get('pixel_crit'))

        # warping criterion
        self.warp_crit = define_criterion(
            self.opt['train'].get('warping_crit'))

        # feature criterion
        self.feat_crit = define_criterion(
            self.opt['train'].get('feature_crit'))
        if self.feat_crit is not None:  # load feature extractor
            feature_layers = self.opt['train']['feature_crit'].get(
                'feature_layers', [8, 17, 26, 35])
            self.net_F = VGGFeatureExtractor(feature_layers).to(self.device)

        # flow & mask criterion
        self.flow_crit = define_criterion(
            self.opt['train'].get('flow_crit'))

        # ping-pong criterion
        self.pp_crit = define_criterion(
            self.opt['train'].get('pingpong_crit'))

        # feature matching criterion
        self.fm_crit = define_criterion(
            self.opt['train'].get('feature_matching_crit'))

        # gan criterion
        self.gan_crit = define_criterion(
            self.opt['train'].get('gan_crit'))


    def train(self, data):
        """ Function for mini-batch training

            Parameters:
                :param data: a batch of training tensor with shape NTCHW
        """

        # ------------ prepare data ------------ #
        lr_data, gt_data = data['lr'], data['gt']

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, gt_h, gt_w = gt_data.size()

        # generate bicubic upsampled data
        bi_data = self.net_G.upsample_func(
            lr_data.view(n * t, c, lr_h, lr_w)).view(n, t, c, gt_h, gt_w)

        # augment data for pingpong criterion
        if self.pp_crit is not None:
            # i.e., (0,1,2,...,t-2,t-1) -> (0,1,2,...,t-2,t-1,t-2,...,2,1,0)
            lr_rev = lr_data.flip(1)[:, 1:, ...]
            gt_rev = gt_data.flip(1)[:, 1:, ...]
            bi_rev = bi_data.flip(1)[:, 1:, ...]

            lr_data = torch.cat([lr_data, lr_rev], dim=1)
            gt_data = torch.cat([gt_data, gt_rev], dim=1)
            bi_data = torch.cat([bi_data, bi_rev], dim=1)


        # ------------ clear optimizers ------------ #
        self.net_G.train()
        self.net_D.train()
        self.optim_G.zero_grad()
        self.optim_D.zero_grad()


        # ------------ forward G ------------ #
        net_G_output_dict = self.net_G.forward_sequence(lr_data)
        hr_data = net_G_output_dict['hr_data']


        # ------------ forward D ------------ #
        for param in self.net_D.parameters():
            param.requires_grad = True

        # feed additional data
        net_D_input_dict = {
            'net_G': self.net_G,
            'lr_data': lr_data,
            'bi_data': bi_data,
            'use_pp_crit': (self.pp_crit is not None),
            'crop_border_ratio': self.opt['train']['discriminator'].get(
                'crop_border_ratio', 1.0)
        }
        net_D_input_dict.update(net_G_output_dict)

        # forward real sequence (gt)
        real_pred, net_D_oputput_dict = self.net_D.forward_sequence(
            gt_data, net_D_input_dict)

        # reuse internal data (e.g., lr optical flow) to reduce computations
        net_D_input_dict.update(net_D_oputput_dict)

        # forward fake sequence (hr)
        fake_pred, _ = self.net_D.forward_sequence(
            hr_data.detach(), net_D_input_dict)


        # ------------ optimize D ------------ #
        self.log_dict = OrderedDict()
        real_pred_D, fake_pred_D = real_pred[0], fake_pred[0]

        # select D update policy
        update_policy = self.opt['train']['discriminator']['update_policy']
        if update_policy == 'adaptive':
            # update D adaptively
            logged_real_pred_D = torch.log(torch.sigmoid(real_pred_D) + 1e-8)
            logged_fake_pred_D = torch.log(torch.sigmoid(fake_pred_D) + 1e-8)

            distance = logged_real_pred_D.mean() - logged_fake_pred_D.mean()

            threshold = self.opt['train']['discriminator']['update_threshold']
            upd_D = distance.item() < threshold
        else:
            upd_D = True

        if upd_D:
            self.cnt_upd_D += 1
            real_loss_D = self.gan_crit(real_pred_D, 1)
            fake_loss_D = self.gan_crit(fake_pred_D, 0)
            loss_D = real_loss_D + fake_loss_D

            # update D
            loss_D.backward()
            self.optim_D.step()
        else:
            loss_D = torch.zeros(1)

        # logging
        self.log_dict['l_gan_D'] = loss_D.item()
        self.log_dict['p_real_D'] = real_pred_D.mean().item()
        self.log_dict['p_fake_D'] = fake_pred_D.mean().item()
        if update_policy == 'adaptive':
            self.log_dict['distance'] = distance.item()
            self.log_dict['n_upd_D'] = self.cnt_upd_D


        # ------------ optimize G ------------ #
        for param in self.net_D.parameters():
            param.requires_grad = False

        # calculate losses
        loss_G = 0
        #
        # pixel (pix) loss
        if self.pix_crit is not None:
            pix_w = self.opt['train']['pixel_crit'].get('weight', 1)
            loss_pix_G = pix_w * self.pix_crit(hr_data, gt_data)
            loss_G += loss_pix_G
            self.log_dict['l_pix_G'] = loss_pix_G.item()
        #
        # warping (warp) loss
        if self.warp_crit is not None:
            lr_curr = net_G_output_dict['lr_curr']
            lr_prev = net_G_output_dict['lr_prev']
            lr_flow = net_G_output_dict['lr_flow']
            lr_warp = net_utils.backward_warp(lr_prev, lr_flow)

            warp_w = self.opt['train']['warping_crit'].get('weight', 1)
            loss_warp_G = warp_w * self.warp_crit(lr_warp, lr_curr)
            loss_G += loss_warp_G
            self.log_dict['l_warp_G'] = loss_warp_G.item()

        # loss_mse_ssim loss
        hr_merge = hr_data.view(-1, c, gt_h, gt_w)
        gt_merge = gt_data.view(-1, c, gt_h, gt_w)

        loss_ms = 1 * loss_mse_ssim_crit(gt_merge, hr_merge)
        loss_G += loss_ms
        self.log_dict['l_ms_G'] = loss_ms.item()

        # # feature (feat) loss
        # if self.feat_crit is not None:
        #     hr_merge = hr_data.view(-1, c, gt_h, gt_w)
        #     gt_merge = gt_data.view(-1, c, gt_h, gt_w)
        #
        #     hr_feat_lst = self.net_F(hr_merge)
        #     gt_feat_lst = self.net_F(gt_merge)
        #     loss_feat_G = 0
        #     for hr_feat, gt_feat in zip(hr_feat_lst, gt_feat_lst):
        #         loss_feat_G += self.feat_crit(hr_feat, gt_feat.detach())
        #
        #     feat_w = self.opt['train']['feature_crit'].get('weight', 1)
        #     loss_feat_G = feat_w * loss_feat_G
        #     loss_G += loss_feat_G
        #     self.log_dict['l_feat_G'] = loss_feat_G.item()
        #
        # # ping-pong (pp) loss
        # if self.pp_crit is not None:
        #     tempo_extent = self.opt['train']['tempo_extent']
        #     hr_data_fw = hr_data[:, :tempo_extent - 1, :, :, :]     # -------->|
        #     hr_data_bw = hr_data[:, tempo_extent:, ...].flip(1) # <--------|
        #     # hr_data_bw = hr_data[:, tempo_extent:, :, :, :]
        #
        #     pp_w = self.opt['train']['pingpong_crit'].get('weight', 1)
        #     loss_pp_G = pp_w * self.pp_crit(hr_data_fw, hr_data_bw)
        #     # loss_pp_G = pp_w * torch.mean(torch.abs(hr_data_fw - hr_data_bw))
        #     loss_G += loss_pp_G
        #     self.log_dict['l_pp_G'] = loss_pp_G.item()
        #
        # feature matching (fm) loss
        if self.fm_crit is not None:
            fake_pred, _ = self.net_D.forward_sequence(hr_data, net_D_input_dict)
            fake_feat_lst, real_feat_lst = fake_pred[-1], real_pred[-1]

            layer_norm = self.opt['train']['feature_matching_crit'].get(
                'layer_norm', [12.0, 14.0, 24.0, 100.0])

            loss_fm_G = 0
            for i in range(len(real_feat_lst)):
                fake_feat, real_feat = fake_feat_lst[i], real_feat_lst[i]
                loss_fm_G += self.fm_crit(
                    fake_feat, real_feat.detach()) / layer_norm[i]

            fm_w = self.opt['train']['feature_matching_crit'].get('weight', 1)
            loss_fm_G = fm_w * loss_fm_G
            loss_G += loss_fm_G
            self.log_dict['l_fm_G'] = loss_fm_G.item()

        # gan loss
        if self.fm_crit is None:
            fake_pred, _ = self.net_D.forward_sequence(hr_data, net_D_input_dict)
        fake_pred_G = fake_pred[0]

        gan_w = self.opt['train']['gan_crit'].get('weight', 1)
        loss_gan_G = gan_w * self.gan_crit(fake_pred_G, True)
        loss_G += loss_gan_G
        # loss_G =  loss_G + loss_gan_G
        self.log_dict['l_gan_G'] = loss_gan_G.item()
        self.log_dict['p_fake_G'] = fake_pred_G.mean().item()

        # update G
        loss_G.backward()
        self.optim_G.step()

    def save(self, current_iter):
        self.save_network(self.net_G, 'G', current_iter)
        self.save_network(self.net_D, 'D', current_iter)
