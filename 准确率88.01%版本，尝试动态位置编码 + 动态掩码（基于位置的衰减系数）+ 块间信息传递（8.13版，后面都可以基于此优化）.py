#版本9，尝试动态位置编码 + 动态掩码（基于位置的衰减系数）+ 块间信息传递(终版)
import json
import math
from dataclasses import dataclass
from typing import Iterable, NamedTuple, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn
from timm.models.vision_transformer import _cfg
from typing import Optional, Union, List


# Device: TypeAlias = str | torch.device | None
Device = Union[str, torch.device, None]



@dataclass
class Mamba2ImageConfig:
    d_model: int              
    n_layer: int = 24
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 32
    image_size: int = 512
    patch_size: int = 16
    num_classes: int = 2
    in_channels: int = 3

    @property
    def seqlen(self) -> int:
        return (self.image_size // self.patch_size) ** 2 + 1

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model    #d_inner =  2 * 1024 = 2048
        assert self.d_inner % self.headdim == 0   #确保头数是整数
        self.nheads = self.d_inner // self.headdim   #计算头数 2048//64 = 32
        self.num_patches = (self.image_size // self.patch_size) ** 2  #计算patch总数


class InferenceCache(NamedTuple):
    '''
    图像分类任务与序列生成任务不同，其主要目标是根据输入图像的整体特征进行类别预测。
    在这种情况下，模型通常不需要逐像素或逐区域地生成输出，而是通过全局特征提取和聚合来完成分类任务。
    因此取消了原mamba2中的用于在自回归推理中存储卷积操作的状态conv_state
    '''
    ssm_state: Tensor # (batch, nheads, headdim, d_state, H, W)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2ImageConfig, H: int, W: int, device: Optional[Union[str, torch.device]] = None):
        return InferenceCache(
            torch.zeros(
                batch_size,
                args.nheads,
                args.headdim,
                args.d_state,
                H,
                W,
                device=device,
            )
        )





#动态位置编码
class DynamicConvPosEnc(nn.Module):
    def __init__(self, dim, grid_size, kernel_size=3):
        super().__init__()
        self.grid_size = grid_size
        self.pos_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                     padding=kernel_size//2, groups=dim),  # 深度可分离卷积
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_size,
                     padding=kernel_size//2, groups=dim)
        )
        # 确保输出维度与输入一致
        self.proj = nn.Linear(dim, dim) if dim % 2 == 0 else nn.Identity()

    def forward(self, x: Tensor):
        """
        x: (batch, num_patches, d_model)
        -> (batch, d_model, H, W)
        -> conv -> (batch, d_model, H, W)
        -> (batch, num_patches, d_model)
        """
        batch, num_patches, d_model = x.shape
        # 转换到2D特征图格式
        x = x.view(batch, *self.grid_size, d_model).permute(0, 3, 1, 2)  # (b,c,h,w)
        # 动态生成位置编码
        pos_enc = self.pos_conv(x)  # (b,c,h,w)
        # 转换回序列格式
        pos_enc = pos_enc.permute(0, 2, 3, 1).view(batch, num_patches, d_model)
        return self.proj(pos_enc)
    

class Mamba2LMHeadModel(nn.Module):
    def __init__(self, args: Mamba2ImageConfig, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.args = args

        #添加可学习的缩放参数
        # self.res_scale = nn.Parameter(torch.ones(()) * 1.0)

        # 特征图的宽和高
        self.grid_size = (
            args.image_size // args.patch_size,
            args.image_size // args.patch_size,
        )

        self.patch_embed = nn.Conv2d(
            in_channels=args.in_channels,
            out_channels=args.d_model,
            kernel_size=args.patch_size,
            stride=args.patch_size,
            padding="valid"
        )

        # ✅ 位置编码仅作用于 patch 序列（不包含 cls_token）
        # self.pos_embed = nn.Parameter(torch.randn(1, args.num_patches, args.d_model))
        self.pos_encoder = DynamicConvPosEnc(
            dim=args.d_model,          # 必须参数1：特征维度
            grid_size=self.grid_size,  # 必须参数2：特征图网格尺寸
            kernel_size=3              # 可选参数：默认3
        )



        self.backbone = nn.ModuleDict(
            dict(
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                mixer=Mamba2(args),  # ✅ 保留 Mamba2 核心结构
                                norm=RMSNorm(args.d_model),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),
                norm_f=RMSNorm(args.d_model),
            )
        )

        self.head = nn.Linear(args.d_model, args.num_classes, bias=False)
        self.layer_scale = nn.Parameter(torch.ones(args.n_layer) * 0.05)

    def forward(self, x: Tensor, h: Optional[List[InferenceCache]] = None, return_h: bool = True) -> Tensor:
        batch_size = x.shape[0]  #(batch_size, in_channels, image_size, image_size) 

        x = self.patch_embed(x)  #(batch_size, d_model, H/patch_size, W/patch_size)
        assert x.shape[2] == self.grid_size[0] and x.shape[3] == self.grid_size[1], \
            f"Grid size mismatch: expected {self.grid_size}, got {x.shape[2:]}"

        x = x.permute(0, 2, 3, 1)  # (batch, H/P, W/P, d_model)
        x = x.view(batch_size, -1, self.args.d_model)  # (batch, num_patches, d_model)  num_patches=image_size//patch_size

        # ✅ 应用位置编码（不包含 cls_token）
        # x = x + self.pos_embed  # 广播机制自动适配 batch_size 维度  # (batch, num_patches, d_model)
        x = x + self.pos_encoder(x)  # 动态位置编码


        new_h = []
        for i, layer in enumerate(self.backbone["layers"]):
            residual = x  # 保存残差连接
            if h is not None and i < len(h):
                x, new_hi = layer["norm"](x, h[i])
            else:
                x, new_hi = layer["mixer"](x, None)

            # 残差连接（修复点）
            x = x + residual * self.layer_scale[i]
            new_h.append(new_hi)

        x = self.backbone["norm_f"](x)

        # ✅ 全局平均池化（GAP）替代 [CLS] token
        h = w = int(x.shape[1] ** 0.5)  # 计算图像块的宽和高
        x = x.view(batch_size, h, w, -1)  # (batch, H, W, d_model)
        x = x.permute(0, 3, 1, 2)       # (batch, d_model, H, W)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # (batch, d_model, 1, 1)
        x = x.flatten(1)                    # (batch, d_model)

        logits = self.head(x)  # (batch, num_classes)
        # return logits, new_h
                # 根据 return_h 参数决定是否返回隐藏状态
        if return_h:
            return logits, new_h
        else:
            return logits  # 评估时只返回logits


class Mamba2(nn.Module):
    def __init__(self, args: Mamba2ImageConfig, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.args = args
        self.device = device




        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=(args.d_conv - 1)//2,
            device=device,
        )

        # self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))
        # self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))
        # self.dt_bias = nn.Parameter(torch.randn(args.nheads, device=device) * 0.01)  # 增大标准差
        # self.A_log = nn.Parameter(torch.randn(args.nheads, device=device) * 0.01)

        # 修改为：
        dt = torch.exp(torch.rand(args.nheads, device=device) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus
        self.dt_bias = nn.Parameter(inv_dt)

        # A_init_range 通常为 (1, 16)
        A = torch.empty(args.nheads, dtype=torch.float32, device=device).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=device))

        # self.D = nn.Parameter(torch.empty(args.nheads, device=device))
        self.D = nn.Parameter(torch.ones(args.nheads, device=device))  # 避免空值
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)
    

    def causal_conv2d(self, x: Tensor) -> Tensor:
        """
        x: (B, C, H, W)
        return: (B, C, H, W)  经过因果卷积后的同尺寸特征图
        """
        B, C, H, W = x.shape
        pad_left = self.args.d_conv - 1          # 仅在左/上侧填充
        pad_top  = self.args.d_conv - 1
        x_pad = F.pad(x, (pad_left, 0, pad_top, 0))   # (left, right, top, bottom)
        out = self.conv2d(x_pad)                       # 卷积
        # 裁剪掉右/下多出来的像素，恢复原始尺寸
        out = out[..., :H, :W]
        return out

    

    def forward(self, u: Tensor, h: Optional[InferenceCache] = None) -> Tensor:    # u(batch, num_patches, d_model)
        
        if h:
            y, h = self.step(u, h)
            return y, h
    
        A = -torch.exp(self.A_log) #状态衰减系数
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.args.d_inner, self.args.d_inner + 2 * self.args.d_state, self.args.nheads],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)
        #xBC [1, 1024, 1152]
        #dt [1, 1024, 16]


        # ✅ 直接使用所有 patch 特征，移除 cls_token 相关逻辑
        patch_features = xBC  # (batch, num_patches, d_inner + 2 * d_state)

        # 计算图像块的宽和高
        h = w = int(patch_features.shape[1] ** 0.5)
        #h/w = 32

        # 调整 patch_features 的形状以符合卷积层输入格式
        patch_features = patch_features.view(-1, h, w, self.args.d_inner + 2 * self.args.d_state)
        #patch_features [1, 32, 32, 1152]
        patch_features = patch_features.permute(0, 3, 1, 2)
        #patch_features [1, 1152, 32, 32]

        # 卷积操作
        patch_features = silu(self.causal_conv2d(patch_features))
        #patch_features [1, 1152, 32, 32]
        patch_features = patch_features.permute(0, 2, 3, 1)
        #patch_features [1, 32, 32, 1152]
        # xBC = patch_features.view(1, -1, patch_features.shape[-1]) 
        xBC = rearrange(patch_features, "b h w D -> b (h w) D") 
        # (1, num_patches, 1152)

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        
        x = rearrange(x, "b seqlen (h_n p) -> b seqlen h_n p", p=self.args.headdim)
        #x [1 1024 16 64]


        y, ssm_state= ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            B,
            C,
            self.args.chunk_size,
            device=self.device,
        )
        y = rearrange(y, "b h w h_n p -> b (h w) h_n p", p=self.args.headdim)

        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b seqlen h_n p -> b seqlen (h_n p)")

        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(ssm_state)

        return y, h





    def step(self, x: Tensor, h: InferenceCache):
        """Take a single inference step for the current input and hidden state"""
        batch, num_patches, d_model = x.shape
        H = W = int(num_patches ** 0.5)

        # 输入投影
        zxbcdt = self.in_proj(x)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.args.d_inner, self.args.d_inner + 2 * self.args.d_state, self.args.nheads],
            dim=-1,
        )

        # 2D 卷积处理
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2)  # (b, C, H, W)
        xBC = self.causal_conv2d(xBC)  # 2D 卷积
        xBC = silu(xBC).permute(0, 2, 3, 1).view(batch, num_patches, -1)

        # 拆分 SSM 参数
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )

        # 将序列数据恢复为2D空间结构
        # 修改后 - 显式拆分第三维度
        x = rearrange(x, "b s (nh p) -> b s nh p", nh=self.args.nheads, p=self.args.headdim)
        x = rearrange(x, "b (h w) nh p -> b h w nh p", h=H, w=W)  # [1,32,32,32,64]
        B = rearrange(B, "b (h w) d -> b h w d", h=H, w=W)        # [1,32,32,64]
        dt = rearrange(dt, "b (h w) nh -> b h w nh", h=H, w=W)    # [1,32,32,32]

        # SSM动态更新（空间感知）
        A = -torch.exp(self.A_log)  # [nheads]
        dt = F.softplus(dt + self.dt_bias)
        dA = torch.exp(dt * A)  # [1,32,32,32] 空间位置感知的衰减系数
        
        # 计算状态更新量（保持空间维度）
        dBx = torch.einsum("b i j h, b i j d, b i j h p -> b h p d i j", dt, B, x)  # [1,32,64,64,32,32]

        
        # 调整dA形状匹配状态 [1,32,1,1,32,32]
        dA = rearrange(dA, "b h w nh -> b nh 1 1 h w") 
        
        # 修改后（正确）
        new_ssm_state = h.ssm_state * dA + dBx
        h = InferenceCache(ssm_state=new_ssm_state)  # 创建新实例替换

        # 输出计算（保持空间结构）
        C = rearrange(C, "b (h w) d -> b h w d", h=H, w=W)  # [1,32,32,64]
        y = torch.einsum("b h p d i j, b i j d -> b i j h p", h.ssm_state, C)  # [1,32,32,32,64]
        y = y + x * self.D.unsqueeze(-1)  # 残差连接
        
        # 恢复序列格式
        y = rearrange(y, "b h w nh p -> b (h w) (nh p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y, h


def segsum_dim(x: Tensor,  device: Optional[Union[str, torch.device]] = None):
    """
    在指定维度上构建半可分转移矩阵
    
    参数:
        x: 输入张量，形状为 (..., D)
        dim: 要构建转移矩阵的维度索引
        device: 计算设备
        
    返回:
        segsum_matrix: 形状 (..., D, D) 的下三角累积和矩阵
    """
    x = torch.clamp(x, min=-1.0, max=1.0)  # 限制输入范围

    dim=1
    # 获取目标维度长度
    T = x.size(dim)
    
    # 将目标维度移到末尾
    x = x.movedim(dim, -1)  # 形状: (..., T)
    
    # 扩展维度构建转移矩阵基础
    x_expanded = repeat(x, "... t -> ... t u", u=T)  # 形状: (..., T, T)
    

    # 创建严格下三角掩码（排除对角线）
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    
    # 应用掩码
    x_masked = x_expanded.masked_fill(~mask, 0)
    
    # 沿目标维度累积求和
    x_segsum = torch.cumsum(x_masked, dim=-2)

    # 添加数值裁剪
    x_segsum = torch.clamp(x_segsum, min=-2.0, max=2.0)


    # 创建对角线包含掩码
    final_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    
    # 应用最终掩码
    segsum_matrix = x_segsum.masked_fill(~final_mask, -torch.inf)

    
    # 将维度移回原位
    return segsum_matrix.movedim(-2, dim).movedim(-1, dim + 1)  #一步到位：把最后两维恢复到原来的 dim 和 dim+1 位置




def ssd(x, A, B, C, chunk_size, device: Optional[Union[str, torch.device]] = None):

    #x [1, 1024, 32, 64](batch, num_patches, nheads, headdim)
    #A [1, 1024, 32](batch, num_patches, nheads)
    #B [1, 1024, 64](batch, num_patches,d_state)
    #C [1, 1024, 64](batch, num_patches,d_state)
    #注意一下头维度和状态维度都是64，不要弄混
    x = rearrange(x, "b (h w) ... -> b h w  ...", h=32, w=32)
    A = rearrange(A, "b (h w) ... -> b h w  ...", h=32, w=32)
    B = rearrange(B, "b (h w) ... -> b h w  ...", h=32, w=32)
    C = rearrange(C, "b (h w) ... -> b h w  ...", h=32, w=32)
    #[1, 32, 32, 16, 64]

    H, W = x.shape[1], x.shape[2]

    assert H % chunk_size == 0 and W % chunk_size == 0

    x = rearrange(x, "b (h c_h) (w c_w) ... -> b h w c_h c_w ...", c_h=chunk_size, c_w=chunk_size)
    A = rearrange(A, "b (h c_h) (w c_w) ... -> b h w c_h c_w ...", c_h=chunk_size, c_w=chunk_size)
    B = rearrange(B, "b (h c_h) (w c_w) ... -> b h w c_h c_w ...", c_h=chunk_size, c_w=chunk_size)
    C = rearrange(C, "b (h c_h) (w c_w) ... -> b h w c_h c_w ...", c_h=chunk_size, c_w=chunk_size)

    Y_row, final_state_row = row_ssd(x, A, B, C, chunk_size, device=device)
    Y_col, final_state_col = col_ssd(Y_row, A, B, C, chunk_size, device=device, initial_states_col=final_state_row)
    Y = rearrange(Y_col, "b h w c_h c_w ... -> b (h c_h) (w c_w) ...")
    return Y, final_state_col  # ✅ 修复：返回两个值（y 和 ssm_state）


def row_ssd(x, A, B, C, chunk_size, device, initial_states_row=None):

    # 前向累积和
    A_forward = rearrange(A, "b h w c_h c_w h_n -> b c_h c_w h w h_n")
    A_cumsum_forward = torch.cumsum(A_forward, dim=1) #A_cumsum_forward形状 ： b c_h c_w h w h_n
    L_forward = torch.exp(segsum_dim(A_cumsum_forward, device)).clamp(max=2.0)

    #L_forward形状为b c_h c_z c_w h w h_n

    # L_forward = rearrange(L_forward, "b h w h_n c_h c_w -> b h w c_h c_w h_n")

    Y_diag_forward = torch.einsum("b h w H W D, b h w H W D, b H Z W h w N, b h w H W N D -> b h w H W N D", C, B, L_forward, x)

    # # 后向累积和
    # A_backward = rearrange(A, "b h w c_h c_w h_n -> b h w h_n c_h c_w")
    # A_cumsum_backward = torch.flip(torch.cumsum(torch.flip(A_backward, dims=[-2]), dim=-2), dims=[-2])
    # L_backward = torch.exp(segsum_2d(A_cumsum_backward, device))
    # L_backward = rearrange(L_backward, "b h w h_n c_h c_w -> b h w c_h c_w h_n")

    # Y_diag_backward = torch.einsum("b h w H W D, b h w H W D, b h w H W N, b h w H W N D -> b h w H W N D", C, B, L_backward, x)


    #A_cumsum_forward shape[1, 8, 8, 32, 4, 4]表示（batch_size, 特征图长上分了几个chunk, 特征图宽上分了几个chunk， 头数， 每个chunk的长， 每个chunk的宽）
    #B shape:torch.Size([1, 8, 8, 4, 4, 64])表示（batch_size, 特征图长上分了几个chunk, 特征图宽上分了几个chunk， 每个chunk的长， 每个chunk的宽， 状态维度）
    #x shape:torch.Size([1, 8, 8, 4, 4, 32, 64])表示（batch_size, 特征图长上分了几个chunk, 特征图宽上分了几个chunk， 每个chunk的长， 每个chunk的宽， 头数， 头维度）
     # 补充：块间状态计算
    decay_states_row = torch.exp(torch.clamp(A_cumsum_forward[:, -1:, -1:, :, :, :] - A_cumsum_forward, max = 3.0))  # 状态衰减因子(b H W h w N)
        # 限制衰减因子范围
    decay_states_row = torch.clamp(decay_states_row, min=0.0, max=1.0)
    states_row = torch.einsum("b h w H W S, b H W h w N, b h w H W N D -> b h w N D S", B, decay_states_row, x)  # 块内状态
    # 添加状态值的裁剪
    states_row = torch.clamp(states_row, min=-2.0, max=2.0)
    #states_row形状：b h w N D S

    # 块间递归传递（类似一维SSD的Step 3）
    if initial_states_row is None:
        initial_states_row = torch.zeros_like(states_row[:, :1]) #在高度方向增加一个维度


    states_row = torch.cat([initial_states_row, states_row], dim=1)  # 沿高度方向拼接初始状态(b, h+1, w, N, D, S)
    A_cumsum_last = A_cumsum_forward[:, -1, :, :, :, :]  # 提取块内最后一个空间位置 (b c_w h w h_n)

    # A_cumsum_last = A_cumsum_last.squeeze(-1).squeeze(-1)  # 压缩 c_h, c_w 维度 -> (b, h, w, h_n)
    A_cumsum_last_padded = F.pad(A_cumsum_last, (0,0, 0,0, 1,0, 0,0, 0,0))  # 在 h 维度前填充 1 个零 -> (b c_w h+1 w h_n)
    A_cumsum_last_padded = rearrange(A_cumsum_last_padded, "b c_w h w h_n -> b h w c_w h_n") #(b h+1 w c_w h_n)

    decay_cumsum = segsum_dim(A_cumsum_last_padded, device=device)  # 沿 h 维度累积(b h+1 z w c_w h_n) [1, 9, 9, 8, 4, 32]

    decay_chunk_row = torch.exp(decay_cumsum)  


    
    
    # 修复后（使用单个字母）
    new_states_row = torch.einsum(
        "b h z w W N, b h w N D S-> b z w N D S",  #z = h+1：表示高度方向块数（含初始状态）
        decay_chunk_row, states_row
    )
    states_row, final_state_row = new_states_row[:, :-1], new_states_row[:, -1]  # 更新状态states_row(b h w N D S), final_state_row(b w N D S)


    # 块间输出计算
    state_decay_out_row = torch.exp(A_cumsum_forward)  #b c_h c_w h w h_n
    Y_off_row = torch.einsum("b h w H W S, b h w N D S, b H W h w N-> b h w H W N D", C, states_row, state_decay_out_row)  # 块间输出

    return Y_diag_forward + Y_off_row, final_state_row  # 合并块内和块间输出


def col_ssd(x, A, B, C, chunk_size, device, initial_states_col=None):
    # 前向累积和
    A_forward = rearrange(A, "b h w c_h c_w h_n -> b c_w c_h h w h_n")
    A_cumsum_forward = torch.cumsum(A_forward, dim=1)
    L_forward = torch.exp(segsum_dim(A_cumsum_forward, device)).clamp(max=2.0) 

    #L_forward形状为b c_w c_z c_h h w h_n 
    # L_forward = rearrange(L_forward, "b h w h_n c_w c_h -> b h w c_h c_w h_n")

    Y_diag_forward = torch.einsum("b h w H W D, b h w H W D, b W Z H h w N, b h w H W N D -> b h w H W N D", C, B, L_forward, x)

    # # 后向累积和
    # A_backward = rearrange(A, "b h w c_h c_w h_n -> b h w h_n c_w c_h")
    # A_cumsum_backward = torch.flip(torch.cumsum(torch.flip(A_backward, dims=[-2]), dim=-2), dims=[-2])
    # L_backward = torch.exp(segsum_2d(A_cumsum_backward, device)).clamp(max=10.0)  
    # L_backward = rearrange(L_backward, "b h w h_n c_w c_h -> b h w c_h c_w h_n")

    # Y_diag_backward = torch.einsum("b h w H W D, b h w H W D, b h w H W N, b h w H W N D -> b h w H W N D", C, B, L_backward, x)



    # 补充：块间状态计算
    decay_states_col = torch.exp(torch.clamp(A_cumsum_forward[:, -1:, -1:, :, :, :] - A_cumsum_forward,max = 2.0))
    decay_states_col = torch.clamp(decay_states_col, min=0.0, max=1.0)
    states_col = torch.einsum("b h w H W S, b H W h w N, b h w H W N D -> b h w N D S", B, decay_states_col, x)  # 块内状态
    states_col = torch.clamp(states_col, min=-3.0, max=3.0)
    # 块间递归传递
    if initial_states_col is None:
        initial_states_col = torch.zeros_like(states_col[:, :, :1])
    else:
        # 假设 states_col 的形状为 (b, h, w, N, D, S)，需将 initial_states_col 扩展为 (b, 1, w, N, D, S)
        initial_states_col = initial_states_col.unsqueeze(1)
        # 获取实际高度块数
        h_size = states_col.size(1)
        initial_states_col = initial_states_col.expand(-1, h_size, -1, -1, -1, -1)  # 扩展 h 维度至 h
        initial_states_col = initial_states_col[:, :, :1]

    states_col = torch.cat([initial_states_col, states_col], dim=2)  # 拼接初始状态

    # 提取块末尾累积值并填充
    A_cumsum_last = A_cumsum_forward[:, -1, :, :, :, :]  # 提取块内最后一个空间位置 (b c_h h w h_n)
    A_cumsum_last_padded = F.pad(A_cumsum_last, (0,0, 1,0, 0,0, 0,0, 0,0))  # 在 h 维度前填充 1 个零 -> (b c_h h w+1 h_n)

    A_cumsum_last_padded = rearrange(A_cumsum_last_padded, "b c_h h w h_n -> b w h c_h h_n") #(b w+1 h c_h h_n)

    decay_cumsum = segsum_dim(A_cumsum_last_padded, device=device)  # 沿 w 维度累积(b w+1 z h c_h h_n) [1, 9, 9, 8, 4, 32]

    decay_chunk_col = torch.exp(decay_cumsum)





    new_states_col = torch.einsum(
        "b w z h H N, b h w N D S-> b h z N D S",  #z = h+1：表示高度方向块数（含初始状态）
        decay_chunk_col, states_col
    )
    states_col, final_state_col = new_states_col[:, :, :-1], new_states_col[:, :, -1]  # 更新状态states_col(b h w N D S), final_state_col(b h N D S)
    final_state_col = final_state_col[:, :1]  #(b N D S)

    # 块间输出计算
    state_decay_out_col = torch.exp(torch.clamp(A_cumsum_forward, max=2.0))  # 显式限制衰减因子
    Y_off_col = torch.einsum("b h w H W S, b h w N D S, b W H h w N -> b h w H W N D", C, states_col, state_decay_out_col)  # 块间输出

    return Y_diag_forward + Y_off_col, final_state_col  # 合并块内和块间输出






# def ssd(x, A, B, C, chunk_size, device: Optional[Union[str, torch.device]] = None):

#     #x [1, 1024, 32, 64](batch, num_patches, nheads, headdim)
#     #A [1, 1024, 32](batch, num_patches, nheads)
#     #B [1, 1024, 64](batch, num_patches,d_state)
#     #C [1, 1024, 64](batch, num_patches,d_state)
#     #注意一下头维度和状态维度都是64，不要弄混
#     x = rearrange(x, "b (h w) ... -> b h w  ...", h=32, w=32)
#     A = rearrange(A, "b (h w) ... -> b h w  ...", h=32, w=32)
#     B = rearrange(B, "b (h w) ... -> b h w  ...", h=32, w=32)
#     C = rearrange(C, "b (h w) ... -> b h w  ...", h=32, w=32)
#     #[1, 32, 32, 16, 64]

#     H, W = x.shape[1], x.shape[2]

#     assert H % chunk_size == 0 and W % chunk_size == 0

#     x = rearrange(x, "b (h c_h) (w c_w) ... -> b h w c_h c_w ...", c_h=chunk_size, c_w=chunk_size)
#     A = rearrange(A, "b (h c_h) (w c_w) ... -> b h w c_h c_w ...", c_h=chunk_size, c_w=chunk_size)
#     B = rearrange(B, "b (h c_h) (w c_w) ... -> b h w c_h c_w ...", c_h=chunk_size, c_w=chunk_size)
#     C = rearrange(C, "b (h c_h) (w c_w) ... -> b h w c_h c_w ...", c_h=chunk_size, c_w=chunk_size)

#     Y_row, final_state_row = row_ssd(x, A, B, C, chunk_size, device=device)
#     Y_col, final_state_col = col_ssd(x, A, B, C, chunk_size, device=device)
#     Y = rearrange(Y_col+Y_row, "b h w c_h c_w ... -> b (h c_h) (w c_w) ...")
#     return Y, final_state_col+final_state_row  # ✅ 修复：返回两个值（y 和 ssm_state）


# def row_ssd(x, A, B, C, chunk_size, device,initial_states_row = None):

#     # 前向累积和
#     A_forward = rearrange(A, "b h w c_h c_w h_n -> b c_h c_w h w h_n")
#     A_cumsum_forward = torch.cumsum(A_forward, dim=1) #A_cumsum_forward形状 ： b c_h c_w h w h_n
#     L_forward = torch.exp(segsum_dim(A_cumsum_forward, device)).clamp(max=3.0)

#     #L_forward形状为b c_h c_z c_w h w h_n

#     # L_forward = rearrange(L_forward, "b h w h_n c_h c_w -> b h w c_h c_w h_n")

#     Y_diag_forward = torch.einsum("b h w H W D, b h w H W D, b H Z W h w N, b h w H W N D -> b h w H W N D", C, B, L_forward, x)



#     #A_cumsum_forward shape[1, 8, 8, 32, 4, 4]表示（batch_size, 特征图长上分了几个chunk, 特征图宽上分了几个chunk， 头数， 每个chunk的长， 每个chunk的宽）
#     #B shape:torch.Size([1, 8, 8, 4, 4, 64])表示（batch_size, 特征图长上分了几个chunk, 特征图宽上分了几个chunk， 每个chunk的长， 每个chunk的宽， 状态维度）
#     #x shape:torch.Size([1, 8, 8, 4, 4, 32, 64])表示（batch_size, 特征图长上分了几个chunk, 特征图宽上分了几个chunk， 每个chunk的长， 每个chunk的宽， 头数， 头维度）
#      # 补充：块间状态计算
#     decay_states_row = torch.exp(torch.clamp(A_cumsum_forward[:, -1:, -1:, :, :, :] - A_cumsum_forward, max = 3.0))  # 状态衰减因子(b H W h w N)
#         # 限制衰减因子范围
#     decay_states_row = torch.clamp(decay_states_row, min=0.0, max=0.5)
#     states_row = torch.einsum("b h w H W S, b H W h w N, b h w H W N D -> b h w N D S", B, decay_states_row, x)  # 块内状态
#     # 添加状态值的裁剪
#     states_row = torch.clamp(states_row, min=-3.0, max=3.0)
#     #states_row形状：b h w N D S

#     # 块间递归传递（类似一维SSD的Step 3）
#     if initial_states_row is None:
#         initial_states_row = torch.zeros_like(states_row[:, :1]) #在高度方向增加一个维度


#     states_row = torch.cat([initial_states_row, states_row], dim=1)  # 沿高度方向拼接初始状态(b, h+1, w, N, D, S)
#     A_cumsum_last = A_cumsum_forward[:, -1, :, :, :, :]  # 提取块内最后一个空间位置 (b c_w h w h_n)

#     # A_cumsum_last = A_cumsum_last.squeeze(-1).squeeze(-1)  # 压缩 c_h, c_w 维度 -> (b, h, w, h_n)
#     A_cumsum_last_padded = F.pad(A_cumsum_last, (0,0, 0,0, 1,0, 0,0, 0,0))  # 在 h 维度前填充 1 个零 -> (b c_w h+1 w h_n)
#     A_cumsum_last_padded = rearrange(A_cumsum_last_padded, "b c_w h w h_n -> b h w c_w h_n") #(b h+1 w c_w h_n)

#     decay_cumsum = segsum_dim(A_cumsum_last_padded, device=device)  # 沿 h 维度累积(b h+1 z w c_w h_n) [1, 9, 9, 8, 4, 32]

#     decay_chunk_row = torch.exp(decay_cumsum)  


    
    
#     # 修复后（使用单个字母）
#     new_states_row = torch.einsum(
#         "b h z w W N, b h w N D S-> b z w N D S",  #z = h+1：表示高度方向块数（含初始状态）
#         decay_chunk_row, states_row
#     )
#     states_row, final_state_row = new_states_row[:, :-1], new_states_row[:, -1]  # 更新状态states_row(b h w N D S), final_state_row(b w N D S)


#     # 块间输出计算
#     state_decay_out_row = torch.exp(A_cumsum_forward)  #b c_h c_w h w h_n
#     Y_off_row = torch.einsum("b h w H W S, b h w N D S, b H W h w N-> b h w H W N D", C, states_row, state_decay_out_row)  # 块间输出

#     return Y_diag_forward + Y_off_row, final_state_row  # 合并块内和块间输出


# def col_ssd(x, A, B, C, chunk_size, device,initial_states_col = None):
#     # 前向累积和
#     A_forward = rearrange(A, "b h w c_h c_w h_n -> b c_w c_h h w h_n")
#     A_cumsum_forward = torch.cumsum(A_forward, dim=1)
#     L_forward = torch.exp(segsum_dim(A_cumsum_forward, device)).clamp(max=3.0) 

#     #L_forward形状为b c_w c_z c_h h w h_n 
#     # L_forward = rearrange(L_forward, "b h w h_n c_w c_h -> b h w c_h c_w h_n")

#     Y_diag_forward = torch.einsum("b h w H W D, b h w H W D, b W Z H h w N, b h w H W N D -> b h w H W N D", C, B, L_forward, x)




#     # 补充：块间状态计算
#     decay_states_col = torch.exp(torch.clamp(A_cumsum_forward[:, -1:, -1:, :, :, :] - A_cumsum_forward,max = 3.0))
#     decay_states_col = torch.clamp(decay_states_col, min=0.0, max=0.5)
#     states_col = torch.einsum("b h w H W S, b H W h w N, b h w H W N D -> b h w N D S", B, decay_states_col, x)  # 块内状态
#     states_col = torch.clamp(states_col, min=-3.0, max=3.0)
#     # 块间递归传递
#     if initial_states_col is None:
#         initial_states_col = torch.zeros_like(states_col[:, :, :1]) #在宽度方向增加一个维度

#     states_col = torch.cat([initial_states_col, states_col], dim=2)  # 拼接初始状态

#     # 提取块末尾累积值并填充
#     A_cumsum_last = A_cumsum_forward[:, -1, :, :, :, :]  # 提取块内最后一个空间位置 (b c_h h w h_n)
#     A_cumsum_last_padded = F.pad(A_cumsum_last, (0,0, 1,0, 0,0, 0,0, 0,0))  # 在 h 维度前填充 1 个零 -> (b c_h h w+1 h_n)

#     A_cumsum_last_padded = rearrange(A_cumsum_last_padded, "b c_h h w h_n -> b w h c_h h_n") #(b w+1 h c_h h_n)

#     decay_cumsum = segsum_dim(A_cumsum_last_padded, device=device)  # 沿 w 维度累积(b w+1 z h c_h h_n) [1, 9, 9, 8, 4, 32]

#     decay_chunk_col = torch.exp(decay_cumsum)  





#     new_states_col = torch.einsum(
#         "b w z h H N, b h w N D S-> b h z N D S",  #z = h+1：表示高度方向块数（含初始状态）
#         decay_chunk_col, states_col
#     )
#     states_col, final_state_col = new_states_col[:, :, :-1], new_states_col[:, :, -1]  # 更新状态states_col(b h w N D S), final_state_col(b h N D S)
#     final_state_col = final_state_col[:, :1]  #(b N D S)

#     # 块间输出计算
#     state_decay_out_col = torch.exp(torch.clamp(A_cumsum_forward, max=6.0))  # 显式限制衰减因子
#     Y_off_col = torch.einsum("b h w H W S, b h w N D S, b W H h w N -> b h w H W N D", C, states_col, state_decay_out_col)  # 块间输出

#     return Y_diag_forward + Y_off_col, final_state_col  # 合并块内和块间输出












class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device) * 1.0)

    def forward(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
        if z is not None:
            x = x * F.silu(z)
        # 使用rsqrt确保数值稳定
        inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight

def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)







def create_mamba2_model(
    d_model=768,              
    n_layer=24,
    d_state=64,
    d_conv=4,
    expand=2,
    headdim=64,
    chunk_size=32,
    image_size=512,
    patch_size=16,
    num_classes=2,
    in_channels=3,
    device=None
):
    args = Mamba2ImageConfig(
        d_model=d_model,
        n_layer=n_layer,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim,
        chunk_size=chunk_size,
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        in_channels=in_channels
    )
    model = Mamba2LMHeadModel(args, device=device)
    return model



# 定义不同版本的模型
def Mamba2_base(device=None):
    model = create_mamba2_model(
        d_model=1024,
        n_layer=4,
        d_state=64,
        d_conv=3,
        expand=2,
        headdim=64,
        chunk_size=4,
        image_size=512,
        patch_size=16,
        num_classes=2,
        in_channels=3,
        device=device
    )
    model.default_cfg = _cfg()
    return model