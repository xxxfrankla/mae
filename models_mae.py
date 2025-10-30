# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
# 文件概览说明：
# 1. 定义了 MaskedAutoencoderViT 类，即 MAE 的完整编码器-解码器结构，骨干是 Vision Transformer。
# 2. 构造函数里创建了图像分块、位置编码、Transformer 编码器与解码器，并初始化参数。
# 3. 提供 patchify/unpatchify/random_masking 等辅助函数，用来把图像转换为 patch 序列并随机掩码。
# 4. forward_encoder/forward_decoder/forward_loss 串起编码、解码和重建损失，最终 forward 返回损失与预测。
# 5. 该实现直接复用了 timm 中的 ViT 模块，是论文《Masked Autoencoders Are Scalable Vision Learners》在 PyTorch 下的核心模型。
from functools import partial  # 工具：用于创建带默认参数的函数（本模块虽然未使用，但与原实现保持一致）

import torch  # PyTorch 主库，负责张量运算与自动微分
import torch.nn as nn  # 神经网络模块，提供 LayerNorm、Linear 等层

from timm.models.vision_transformer import PatchEmbed, Block  # ViT 中的图像分块模块与 Transformer 块

from util.pos_embed import get_2d_sincos_pos_embed  # 仓库内函数，生成二维正弦余弦位置编码


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """  # 声明类：继承 nn.Module，表示 MAE 模型的整体结构，编码器/解码器都基于 ViT
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()  # 调用父类构造函数，初始化基础 nn.Module 结构

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)  # 将图像划分为 patch 并线性投影到 embed_dim 维度
        num_patches = self.patch_embed.num_patches  # 记录每张图像被切成的 patch 数量（H/p * W/p）

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 可学习的分类 token，放在序列开头
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding  # ViT 位置编码，长度比 patch 数多 1（给 cls_token 用），冻结不训练

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])  # 构建编码器的 Transformer 堆叠：depth 个 Block
        self.norm = norm_layer(embed_dim)  # 编码器输出的 LayerNorm，通常是 LayerNorm
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)  # 先把编码器输出的维度映射到解码器维度

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))  # 被遮挡 patch 的占位 token，可学习

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding  # 解码器使用的固定位置编码

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])  # 解码器的 Transformer 堆叠：depth 可以比编码器浅

        self.decoder_norm = norm_layer(decoder_embed_dim)  # 解码器输出的 LayerNorm
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch  # 将每个 token 预测回原始 patch 的像素（p*p*3）
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss  # 标记是否在损失里使用归一化像素（论文中可选）

        self.initialize_weights()  # 调用自定义初始化函数，设置位置编码和参数初值

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)  # 生成编码器的二维正弦余弦位置编码
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))  # 将 numpy 数组拷贝进参数张量，并保持 requires_grad=False

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)  # 同理生成解码器的固定位置编码
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))  # 拷贝并冻结

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data  # 取出 patch embedding 中的卷积核权重
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))  # 按照线性层的方式做 Xavier 均匀初始化，稳定训练

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)  # 将 cls token 初始化为均值 0、方差 0.02 的正态分布
        torch.nn.init.normal_(self.mask_token, std=.02)  # mask token 同样初始化

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)  # 遍历模型的每一层，调用自定义权重初始化函数

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  # 若当前层是线性层
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)  # Xavier 均匀初始化权重，保持输出方差稳定
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为 0
        elif isinstance(m, nn.LayerNorm):  # 若当前层是 LayerNorm
            nn.init.constant_(m.bias, 0)  # LayerNorm 的偏置初始化为 0
            nn.init.constant_(m.weight, 1.0)  # 缩放系数初始化为 1，保持输入分布

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]  # patch 的边长（例如 16）
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0  # 断言输入图像为正方形且能被 patch 尺寸整除

        h = w = imgs.shape[2] // p  # 图像在纵横方向各切成多少块
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))  # 先重排维度：把每个 patch 的像素分组
        x = torch.einsum('nchpwq->nhwpqc', x)  # 调整维度顺序：把 patch 收集到一起
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))  # 将每个 patch 展平成一维向量，得到序列表示
        return x  # 返回尺寸 [batch, patch数, patch像素向量长度]

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]  # patch 边长
        h = w = int(x.shape[1]**.5)  # patch 个数开根号得到每行/列 patch 数
        assert h * w == x.shape[1]  # 确保 patch 数量是完美平方（因为图像是正方形）
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))  # 先还原每个 patch 的二维像素块
        x = torch.einsum('nhwpqc->nchpwq', x)  # 调整维度，把通道放到第二维
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))  # 将所有 patch 拼接回完整图像
        return imgs  # 返回重建后的图像张量

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))  # 计算需要保留的 patch 数量
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 记录逆排序索引，用于将顺序还原

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # 取排序后最小的几个索引（对应保留的 patch）
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # 把保留的 patch 提取出来

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)  # 先整体置为 1（表示遮挡）
        mask[:, :len_keep] = 0  # 前 len_keep 个位置改为 0（表示保留）
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)  # 按原顺序还原掩码，得到每个位置的保留/遮挡标记

        return x_masked, mask, ids_restore  # 返回保留的 patch、二值掩码和恢复顺序索引

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)  # 图像分块并线性投影，形状变为 [N, patch数, embed_dim]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]  # 给每个 patch 加上固定的正弦余弦位置编码（不含 cls 位置）

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)  # 随机遮挡大部分 patch，返回保留的序列及掩码

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # 给 cls token 加上对应的位置编码
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # 扩展到 batch 大小
        x = torch.cat((cls_tokens, x), dim=1)  # 将 cls token 拼到序列最前面

        # apply Transformer blocks
        for blk in self.blocks:  # 依次通过编码器的 Transformer block
            x = blk(x)
        x = self.norm(x)  # 最后做 LayerNorm 对特征做归一化

        return x, mask, ids_restore  # 返回编码器输出、掩码以及还原索引

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)  # 先把编码器输出映射到解码器的特征维度

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)  # 根据缺失的 patch 数量生成 mask token
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token  # 取出除 cls 外的 token，并拼上 mask token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle  # 按照原始 patch 顺序还原序列
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token  # 把 cls token 再拼回序列开头

        # add pos embed
        x = x + self.decoder_pos_embed  # 加上解码器使用的位置编码

        # apply Transformer blocks
        for blk in self.decoder_blocks:  # 逐层通过解码器的 Transformer block
            x = blk(x)
        x = self.decoder_norm(x)  # 最后做 LayerNorm

        # predictor projection
        x = self.decoder_pred(x)  # 将每个 token 映射成原始 patch 的像素值

        # remove cls token
        x = x[:, 1:, :]  # 去掉 cls token，只保留 patch 重建结果

        return x  # 返回重建后的 patch 序列

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)  # 将原图像转换成 patch 序列作为重建目标
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)  # 计算每个 patch 的像素均值
            var = target.var(dim=-1, keepdim=True)  # 计算每个 patch 的像素方差
            target = (target - mean) / (var + 1.e-6)**.5  # 按 patch 标准化，避免亮度差异影响

        loss = (pred - target) ** 2  # 逐像素计算 MSE
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch  # 对每个 patch 内的像素取平均，得到 patch 级损失

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches  # 只在被遮挡的 patch 上求均值，符合 MAE 目标
        return loss  # 返回标量损失

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)  # 先经过编码器得到潜在表示及掩码
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]  # 用解码器重建完整 patch 序列
        loss = self.forward_loss(imgs, pred, mask)  # 计算重建损失
        return loss, pred, mask  # 返回损失、预测的 patch 以及掩码，用于训练与可视化


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)  # 构建 ViT-Base 规模的 MAE，解码器 512 维 8 层
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)  # 构建 ViT-Large 规模的 MAE
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)  # 构建 ViT-Huge 规模的 MAE（patch14）
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks  # 给常用架构设置别名，方便在配置里引用
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
