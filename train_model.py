import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
import random
import pennylane as qml
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.init as init
import os

# ==================== 字体设置 ====================
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 路径设置 ====================
ROOT_DIR = r"D:\DALUNWEN"
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")
PRETRAINED_PATH = os.path.join(ROOT_DIR, "pretrained", "medical_shadow_pretrained.pth")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output", "A1")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
LOG_PATH = os.path.join(OUTPUT_DIR, "training_log.csv")

# 创建输出目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


# ==================== 随机种子设置 ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()


# ==================== CBAM结构 ====================
class CBAM(nn.Module):
    """卷积块注意力模块：通道注意力+空间注意力"""

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        # 1. 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化获取通道特征
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()  # 输出通道权重（0-1）
        )

        # 2. 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()  # 输出空间权重（0-1）
        )

    def forward(self, x):
        # 通道注意力加权
        channel_weight = self.channel_attn(x)
        x = x * channel_weight  # 逐通道加权

        # 空间注意力加权（基于通道池化特征）
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道最大
        spatial_map = torch.cat([avg_out, max_out], dim=1)  # 拼接为2通道
        spatial_weight = self.spatial_attn(spatial_map)
        x = x * spatial_weight  # 逐空间位置加权
        return x


# ==================== 量子特征蒸馏器 ====================
class QuantumDistiller(nn.Module):
    """量子特征蒸馏器：量子比特纠缠层+动态门控"""

    def __init__(self, in_channels, out_channels, pretrained=None, gate_threshold=0.05, debug=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_qubits = min(8, in_channels)  # 量子比特数根据输入通道数动态调整（不超过8）
        self.gate_threshold = gate_threshold
        self.debug = debug  # 控制调试输出
        self.debug_counter = 0

        # 经典预处理
        self.preprocess = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_qubits)
        )

        # 经典后处理
        self.postprocess = nn.Sequential(
            nn.Linear(self.n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

        # 局部特征转换
        self.local_transform = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 量子电路初始化（支持批处理）
        self.quantum_circuit = self.create_quantum_circuit()
        self.quantum_weights = nn.Parameter(0.1 * torch.randn(3 * self.n_qubits))  # 量子门参数
        self.channel_weights = nn.Parameter(torch.ones(in_channels) / in_channels)  # 通道权重

        # 加载预训练权重
        if pretrained and os.path.exists(pretrained):
            try:
                self.load_pretrained_weights(pretrained)
            except Exception as e:
                print(f"警告：加载预训练权重失败 {pretrained}，错误: {e}")
                print("将从头训练量子层")
        elif pretrained:
            print(f"警告：预训练权重文件不存在 {pretrained}，将从头训练量子层")

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def create_quantum_circuit(self):
        """量子比特纠缠电路（支持批处理输入）"""
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # 输入编码（支持批处理）
            for i in range(self.n_qubits):
                qml.RY(np.pi * inputs[:, i], wires=i)  # 注意这里的[:, i]处理批处理

            # 变分层1
            for i in range(self.n_qubits):
                qml.RY(weights[i], wires=i)

            # 纠缠层
            for i in range(0, self.n_qubits - 1, 2):
                qml.CNOT(wires=[i, i + 1])

            # 变分层2
            for i in range(self.n_qubits):
                qml.RY(weights[self.n_qubits + i], wires=i)

            # 测量
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def load_pretrained_weights(self, pretrained_path):
        """加载量子层预训练权重"""
        state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        quantum_state_dict = {k: v for k, v in state_dict.items() if 'quantum' in k or 'channel_weights' in k}
        self.load_state_dict(quantum_state_dict, strict=False)
        print(f"已加载预训练权重：{pretrained_path}")

    def forward(self, x):
        batch_size, _, H, W = x.shape

        # 全局特征提取
        x_global = torch.mean(x, dim=(2, 3))  # [B, C]

        # 量子特征处理（批处理优化）
        weighted_input = x_global * self.channel_weights  # 通道加权 [B, C]
        x_quantum = self.quantum_circuit(weighted_input, self.quantum_weights)  # [n_qubits, B]
        x_quantum = torch.stack(x_quantum).T.float()  # 转置为 [B, n_qubits]

        # 动态门控
        quantum_strength = torch.norm(x_quantum, p=2, dim=1, keepdim=True)  # [B, 1]
        gate = torch.sigmoid(quantum_strength - self.gate_threshold)  # 使用sigmoid平滑门控

        # 量子特征扩展到空间维度
        x_global_processed = self.postprocess(x_quantum)  # [B, out_channels]
        x_global_spatial = x_global_processed.view(batch_size, self.out_channels, 1, 1)
        x_global_spatial = nn.functional.adaptive_avg_pool2d(x_global_spatial, (H, W))

        # 局部特征转换
        x_local = self.local_transform(x)  # [B, out_channels, H, W]

        # 记录监控指标
        self.quantum_strength = quantum_strength.mean().item()
        self.gate_activation = gate.mean().item()

        # 确保门控维度与特征图匹配
        if gate.dim() == 2:
            gate = gate.unsqueeze(-1).unsqueeze(-1)  # [B, 1] -> [B, 1, 1, 1]

        # 确保尺寸匹配
        if x_local.size() != x_global_spatial.size():
            x_global_spatial = nn.functional.interpolate(
                x_global_spatial, size=x_local.size()[2:], mode='bilinear', align_corners=False
            )

        # 调试输出（可控）
        if self.debug and self.debug_counter % 100 == 0:
            print(
                f"量子蒸馏器调试: 输入尺寸={x.shape}, 局部特征尺寸={x_local.shape}, 全局特征尺寸={x_global_spatial.shape}, 门控尺寸={gate.shape}")
        self.debug_counter += 1

        return x_local + gate * x_global_spatial, x_quantum, quantum_strength


# ==================== 量子特征映射器（新增） ====================
class QuantumFeatureMapper(nn.Module):
    """量子特征映射器：通过经典计算模拟量子特性，实现稳定的特征增强"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1. 量子态映射（模拟叠加态）
        self.superposition = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Tanh()  # 限制特征范围至[-1,1]，模拟量子态振幅
        )

        # 2. 纠缠关联（模拟量子纠缠，强化空间关联性）
        self.entanglement = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2, groups=8),  # 分组卷积模拟局部纠缠
            nn.BatchNorm2d(64),
            nn.Tanh()
        )

        # 3. 经典投影（模拟量子测量）
        self.projection = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=1)  # 模拟测量概率分布
        )

        # 4. 残差连接（控制量子特征贡献度）
        self.residual_scale = nn.Parameter(torch.tensor(0.2))  # 固定初始权重0.2

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化卷积层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: 输入特征图 [B, C, H, W]

        # 1. 模拟量子叠加态（特征维度扩展与归一化）
        x_super = self.superposition(x)  # [B, 64, H, W]

        # 2. 模拟量子纠缠（强化空间局部关联）
        x_entangle = self.entanglement(x_super)  # [B, 64, H, W]

        # 3. 模拟量子测量（投影回经典特征空间）
        quantum_feature = self.projection(x_entangle)  # [B, out_channels, H, W]

        # 4. 残差融合（固定权重控制贡献度）
        if x.size(1) != self.out_channels:
            x = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)(x)  # 特征维度匹配
        fused_feature = x + self.residual_scale * quantum_feature  # 残差连接

        return {
            'fused': fused_feature,
            'quantum_feature': quantum_feature  # 中间量子特征（用于可视化）
        }


# ==================== UNet输出卷积 ====================
class OutConv(nn.Module):
    """输出卷积层，将特征映射到目标类别数"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ==================== UNet骨干网络 ====================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 根据通道数动态调整CBAM的reduction_ratio
        reduction_ratio = max(4, out_channels // 32)  # 避免过小的压缩比
        self.cbam = CBAM(out_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.cbam(x)  # 应用卷积块内的CBAM
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)  # 分离出maxpool以便添加CBAM
        # 根据通道数动态调整CBAM参数
        reduction_ratio = max(4, in_channels // 32)
        self.cbam_after_pool = CBAM(in_channels, reduction_ratio=reduction_ratio)
        # 卷积块（已包含CBAM）
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.cbam_after_pool(x)  # 下采样后添加CBAM
        x = self.conv(x)  # 卷积块内已有CBAM
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            # 动态调整CBAM参数
            cbam_channels = in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            # 动态调整CBAM参数
            cbam_channels = in_channels // 2

        reduction_ratio = max(4, cbam_channels // 32)
        self.cbam_after_up = CBAM(cbam_channels, reduction_ratio=reduction_ratio)

        # 解码器额外增强CBAM
        final_reduction = max(4, out_channels // 32)
        self.cbam_final = CBAM(out_channels, reduction_ratio=final_reduction)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.cbam_after_up(x1)  # 上采样后添加CBAM

        # 处理尺寸差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)  # 卷积块内已有CBAM
        x = self.cbam_final(x)  # 最终CBAM增强
        return x


class QuantumEnhancedUNet(nn.Module):
    """量子增强UNet，在解码器每一步添加QuantumFeatureMapper"""

    def __init__(self, n_channels=3, n_classes=1, bilinear=True, pretrained=None, debug=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.debug = debug

        # 编码器 - 每层都包含CBAM
        self.inc = DoubleConv(n_channels, 64)  # 包含CBAM
        self.down1 = Down(64, 128)  # 包含2个CBAM
        self.down2 = Down(128, 256)  # 包含2个CBAM
        self.down3 = Down(256, 512)  # 包含2个CBAM
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # 包含2个CBAM

        # 量子特征蒸馏器
        self.quantum_distiller = QuantumDistiller(
            in_channels=1024 // factor,
            out_channels=1024 // factor,
            pretrained=pretrained,
            debug=debug
        )

        # 解码器 - 每层都包含CBAM
        self.up1 = Up(1024, 512 // factor, bilinear)  # 包含3个CBAM
        self.up2 = Up(512, 256 // factor, bilinear)  # 包含3个CBAM
        self.up3 = Up(256, 128 // factor, bilinear)  # 包含3个CBAM
        self.up4 = Up(128, 64, bilinear)  # 包含3个CBAM

        # 在解码器每一步添加QuantumFeatureMapper（新增部分）
        self.mapper1 = QuantumFeatureMapper(in_channels=512 // factor, out_channels=512 // factor)
        self.mapper2 = QuantumFeatureMapper(in_channels=256 // factor, out_channels=256 // factor)
        self.mapper3 = QuantumFeatureMapper(in_channels=128 // factor, out_channels=128 // factor)
        self.mapper4 = QuantumFeatureMapper(in_channels=64, out_channels=64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.debug:
            print(f"UNet编码路径: x1={x1.shape}, x2={x2.shape}, x3={x3.shape}, x4={x4.shape}, x5={x5.shape}")

        # 量子特征蒸馏
        x5, x_quantum, quantum_strength = self.quantum_distiller(x5)

        # 解码路径（每一步添加QuantumFeatureMapper）
        x = self.up1(x5, x4)
        x = self.mapper1(x)['fused']  # 解码器第一步添加映射器

        x = self.up2(x, x3)
        x = self.mapper2(x)['fused']  # 解码器第二步添加映射器

        x = self.up3(x, x2)
        x = self.mapper3(x)['fused']  # 解码器第三步添加映射器

        x = self.up4(x, x1)
        x = self.mapper4(x)['fused']  # 解码器第四步添加映射器

        logits = self.outc(x)

        return {
            'logits': logits,
            'quantum_features': x_quantum,
            'quantum_strength': quantum_strength,
            'gate_activation': self.quantum_distiller.gate_activation
        }


# ==================== 数据预处理与加载 ====================
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        img = img.resize(self.size, Image.BICUBIC)
        mask = mask.resize(self.size, Image.NEAREST)
        return img, mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask


class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        img = img.rotate(angle, Image.BICUBIC)
        mask = mask.rotate(angle, Image.NEAREST)
        return img, mask


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        img = img.crop((j, i, j + tw, i + th))
        mask = mask.crop((j, i, j + tw, i + th))

        return img, mask


class ToTensor:
    def __call__(self, img, mask):
        img = transforms.functional.to_tensor(img)
        mask = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)
        mask = torch.clamp(mask, 0, 1)
        return img, mask


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img = transforms.functional.normalize(img, self.mean, self.std)
        return img, mask


class ShadowDataset(Dataset):
    """改进的数据集加载器，支持跳过损坏文件"""

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root_dir, split, "images")
        self.masks_dir = os.path.join(root_dir, split, "masks")

        print(f"检查数据集路径: {self.images_dir}")
        print(f"检查掩码路径: {self.masks_dir}")

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"掩码目录不存在: {self.masks_dir}")

        # 加载并验证有效图像
        self.valid_images = []
        image_files = sorted([f for f in os.listdir(self.images_dir)
                              if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))])
        mask_files = sorted([f for f in os.listdir(self.masks_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # 构建图像和掩码的基础名称映射
        image_bases = {os.path.splitext(f)[0]: f for f in image_files}
        mask_bases = {os.path.splitext(f)[0]: f for f in mask_files}

        # 验证并收集有效文件对
        for base_name in image_bases.keys() & mask_bases.keys():
            img_file = image_bases[base_name]
            mask_file = mask_bases[base_name]

            # 验证图像文件是否有效
            img_path = os.path.join(self.images_dir, img_file)
            mask_path = os.path.join(self.masks_dir, mask_file)

            try:
                # 快速验证图像是否可以打开
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(mask_path) as mask:
                    mask.verify()

                self.valid_images.append((img_file, mask_file))
            except Exception as e:
                print(f"跳过无效文件对: {img_file}, {mask_file}, 错误: {e}")

        print(f"{split} 集加载完成，有效图像-掩码对: {len(self.valid_images)}")

        if len(self.valid_images) == 0:
            raise ValueError(f"{split} 集无有效图像-掩码对")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        try:
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            # 应用数据转换
            if self.transform is not None:
                img, mask = self.transform(img, mask)

            # 验证数据类型
            if not isinstance(img, torch.Tensor):
                img = transforms.functional.to_tensor(img)

            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(np.array(mask, dtype=np.float32)).unsqueeze(0)

            # 确保掩码值在0-1范围内
            mask = torch.clamp(mask, 0, 1)

            return img, mask

        except Exception as e:
            print(f"警告：加载文件失败: 图像={img_name}, 掩码={mask_name}, 错误={e}")
            # 替换为随机有效样本而非返回None
            fallback_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(fallback_idx)


# ==================== 评价指标 ====================
def calculate_iou(pred, target, threshold=0.5):
    """计算IoU"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def calculate_dice(pred, target, threshold=0.5):
    """计算Dice系数"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6)
    return dice.mean().item()


def calculate_metrics(pred, target, threshold=0.5):
    """计算多种评价指标"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    # 计算真阳性、假阳性、真阴性、假阴性
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()

    # 计算精确率、召回率和F1分数
    precision = (tp + 1e-8) / (tp + fp + 1e-8)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


# ==================== 损失函数 ====================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, pred, target):
        return self.bce_loss(pred, target)


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        target = torch.clamp(target, 0, 1)
        return self.dice_weight * self.dice_loss(pred, target) + self.bce_weight * self.bce_loss(pred, target)


# ==================== 量子特征热力图可视化 ====================
def visualize_quantum_heatmap(model, dataloader, device, save_dir, epoch, num_samples=5):
    """可视化量子特征热力图（优化为CVPR风格：关联分割贡献）"""
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            logits = outputs['logits']
            quantum_features = outputs['quantum_features']

            for j in range(min(images.size(0), 2)):  # 每个批次最多可视化2个样本
                # 数据准备
                img = images[j].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())  # 归一化
                mask = masks[j].cpu().squeeze().numpy()
                pred = torch.sigmoid(logits[j]).cpu().squeeze().numpy()
                q_features = quantum_features[j].cpu().numpy()

                # 热力图生成
                H, W = mask.shape  # 获取掩码尺寸（与图像一致）
                q_heatmap = np.zeros((H, W), dtype=np.float32)  # 初始化二维热力图

                # 热力图关联分割贡献
                # 1. 量子特征强度归一化（0~1）
                q_feat_norm = (q_features - q_features.min()) / (q_features.max() - q_features.min() + 1e-8)
                avg_q_strength = q_feat_norm.mean()  # 量子特征平均强度（量化贡献）

                # 2. 分割预测置信度（0~1，越高说明预测越可信）
                pred_confidence = pred  # 直接使用模型预测掩码的置信度

                # 3. 热力图 = 量子特征强度 × 预测置信度（关联任务：量子对可信预测的贡献）
                for i_pixel in range(H):
                    for j_pixel in range(W):
                        q_heatmap[i_pixel, j_pixel] = avg_q_strength * pred_confidence[i_pixel, j_pixel]

                # 4. 避开掩码区域（仅显示背景中量子特征的作用）
                q_heatmap[mask >= 0.5] = 0

                # 处理无效量子特征
                if np.isnan(q_heatmap).any() or (q_heatmap.max() - q_heatmap.min() < 1e-6):
                    q_heatmap = np.zeros_like(q_heatmap)
                    print(f"样本 {i * 2 + j} 量子特征无效，已使用零矩阵替代")
                else:
                    # 归一化
                    q_min, q_max = q_heatmap.min(), q_heatmap.max()
                    q_heatmap = (q_heatmap - q_min) / (q_max - q_min)

                # 避开掩码区域
                masked_heatmap = np.ma.masked_where(mask < 0.5, q_heatmap)

                # 绘图逻辑
                fig, axes = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1.2]})

                # 原始图像
                axes[0].imshow(img)
                axes[0].set_title("原始图像", fontsize=14)
                axes[0].axis('off')

                # 真实掩码
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title("真实掩码", fontsize=14)
                axes[1].axis('off')

                # 预测掩码
                axes[2].imshow(pred, cmap='viridis')
                axes[2].set_title("预测掩码", fontsize=14)
                axes[2].axis('off')

                # 量子热力图
                axes[3].imshow(img)
                # CVPR风格配色
                quantum_cmap = LinearSegmentedColormap.from_list(
                    "cvpr_heatmap",
                    [(0.0, 0.0, 1.0),  # 低强度（蓝）
                     (0.0, 1.0, 1.0),  # 中低（青）
                     (1.0, 1.0, 0.0),  # 中高（黄）
                     (1.0, 0.0, 0.0)]  # 高强度（红）
                )
                heatmap_im = axes[3].imshow(masked_heatmap, cmap=quantum_cmap, alpha=0.6)
                # 标题添加量化指标
                axes[3].set_title(f"量子特征热力图（贡献度：{avg_q_strength:.2f}）", fontsize=14)
                axes[3].axis('off')

                # 颜色条
                cbar = fig.colorbar(heatmap_im, ax=axes[3], orientation='vertical', pad=0.02)
                cbar.set_label('量子特征叠加强度', fontsize=12)
                cbar.ax.tick_params(labelsize=10)

                plt.tight_layout()
                plt.savefig(
                    os.path.join(save_dir, f"quantum_heatmap_epoch_{epoch}_sample_{i * 2 + j}.png"),
                    dpi=300, bbox_inches='tight'
                )
                plt.close()


# ==================== 学习率预热 ====================
class LRSchedulerWithWarmup:
    """学习率调度器，带预热阶段"""

    def __init__(self, optimizer, warmup_epochs=5, total_epochs=100, start_lr=1e-5, max_lr=1e-3, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_epoch = 0

        # 初始化学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.start_lr

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # 预热阶段：线性增加学习率
            lr = self.start_lr + (self.max_lr - self.start_lr) * (self.current_epoch / self.warmup_epochs)
        else:
            # 预热后：余弦退火衰减
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        # 更新学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


# ==================== 训练函数 ====================
def train_model(debug=False):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 定义数据变换
    train_transform = Compose([
        Resize((224, 224)),  # 统一调整为224x224
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=15),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = Compose([
        Resize((224, 224)),  # 统一调整为224x224
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    print("加载训练集...")
    train_dataset = ShadowDataset(root_dir=DATASET_DIR, split="train", transform=train_transform)
    print("加载验证集...")
    val_dataset = ShadowDataset(root_dir=DATASET_DIR, split="val", transform=val_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = QuantumEnhancedUNet(
        n_channels=3,
        n_classes=1,
        pretrained=PRETRAINED_PATH,
        debug=debug
    ).to(device)

    # 定义损失函数和优化器
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 学习率调度器（带预热）
    scheduler = LRSchedulerWithWarmup(
        optimizer,
        warmup_epochs=5,
        total_epochs=100,
        start_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6
    )

    # 记录最佳模型
    best_val_iou = 0.0
    best_epoch = 0
    # 早停机制
    patience = 15
    no_improve_epochs = 0

    # 创建日志文件
    with open(LOG_PATH, 'w') as f:
        f.write(
            "Epoch,Train Loss,Train IoU,Train Dice,Val Loss,Val IoU,Val Dice,Val Precision,Val Recall,Val F1,Quantum Strength,Gate Activation,Learning Rate\n")

    # 训练循环
    for epoch in range(1, 101):  # 100个epochs
        print(f"\n===== Epoch {epoch}/{100} =====")

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0

        quantum_strengths = []
        gate_activations = []

        train_progress = tqdm(train_loader, desc="训练中")
        for images, masks in train_progress:
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            logits = outputs['logits']
            loss = criterion(logits, masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 记录指标
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            train_iou += calculate_iou(logits, masks) * batch_size
            train_dice += calculate_dice(logits, masks) * batch_size

            # 记录量子特征强度和门控激活
            quantum_strengths.append(outputs['quantum_strength'].mean().item())
            gate_activations.append(outputs['gate_activation'])

            # 更新进度条
            train_progress.set_postfix({
                'loss': loss.item(),
                'iou': calculate_iou(logits, masks),
                'dice': calculate_dice(logits, masks),
                'q_strength': outputs['quantum_strength'].mean().item(),
                'gate': outputs['gate_activation']
            })

        # 计算平均指标
        train_loss /= len(train_dataset)
        train_iou /= len(train_dataset)
        train_dice /= len(train_dataset)
        avg_quantum_strength = sum(quantum_strengths) / len(quantum_strengths) if quantum_strengths else 0
        avg_gate_activation = sum(gate_activations) / len(gate_activations) if gate_activations else 0

        print(
            f"训练: Loss={train_loss:.4f}, IoU={train_iou:.4f}, Dice={train_dice:.4f}, 量子强度={avg_quantum_strength:.4f}, 门控激活={avg_gate_activation:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0

        val_progress = tqdm(val_loader, desc="验证中")
        with torch.no_grad():
            for images, masks in val_progress:
                images = images.to(device)
                masks = masks.to(device)

                # 前向传播
                outputs = model(images)
                logits = outputs['logits']
                loss = criterion(logits, masks)

                # 计算指标
                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                val_iou += calculate_iou(logits, masks) * batch_size
                val_dice += calculate_dice(logits, masks) * batch_size

                # 计算详细指标
                metrics = calculate_metrics(logits, masks)
                val_precision += metrics['precision'] * batch_size
                val_recall += metrics['recall'] * batch_size
                val_f1 += metrics['f1'] * batch_size

                # 更新进度条
                val_progress.set_postfix({
                    'loss': loss.item(),
                    'iou': calculate_iou(logits, masks),
                    'dice': calculate_dice(logits, masks)
                })

        # 计算平均指标
        val_loss /= len(val_dataset)
        val_iou /= len(val_dataset)
        val_dice /= len(val_dataset)
        val_precision /= len(val_dataset)
        val_recall /= len(val_dataset)
        val_f1 /= len(val_dataset)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"验证: Loss={val_loss:.4f}, IoU={val_iou:.4f}, Dice={val_dice:.4f}, Precision={val_precision:.4f}, Recall={val_recall:.4f}, F1={val_f1:.4f}, LR={current_lr:.6f}")

        # 学习率调整
        scheduler.step()

        # 保存日志
        with open(LOG_PATH, 'a') as f:
            f.write(
                f"{epoch},{train_loss},{train_iou},{train_dice},{val_loss},{val_iou},{val_dice},{val_precision},{val_recall},{val_f1},{avg_quantum_strength},{avg_gate_activation},{current_lr}\n")

        # 保存最佳模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_model.pth"))
            print(f"保存最佳模型: Epoch {epoch}, Val IoU: {val_iou:.4f}")
            no_improve_epochs = 0  # 重置早停计数器
        else:
            no_improve_epochs += 1
            print(f"早停计数器: {no_improve_epochs}/{patience}")
            if no_improve_epochs >= patience:
                print(f"早停于Epoch {epoch}，验证指标不再提升")
                break

        # 保存当前模型
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"model_epoch_{epoch}.pth"))

        # 可视化量子特征
        if epoch % 10 == 0:  # 每10个epoch可视化一次
            visualize_quantum_heatmap(model, val_loader, device, VISUALIZATION_DIR, epoch)

    print(f"训练完成！最佳模型在第 {best_epoch} 个epoch，验证IoU为 {best_val_iou:.4f}")

    # 加载最佳模型并在测试集上评估
    checkpoint = torch.load(os.path.join(MODEL_DIR, f"best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 准备测试集
    print("加载测试集...")
    test_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ShadowDataset(root_dir=DATASET_DIR, split="test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print("在测试集上评估...")
    test_loss = 0.0
    test_iou = 0.0
    test_dice = 0.0
    test_precision = 0.0
    test_recall = 0.0
    test_f1 = 0.0

    test_progress = tqdm(test_loader, desc="测试中")
    with torch.no_grad():
        for images, masks in test_progress:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            logits = outputs['logits']
            loss = criterion(logits, masks)

            batch_size = images.size(0)
            test_loss += loss.item() * batch_size
            test_iou += calculate_iou(logits, masks) * batch_size
            test_dice += calculate_dice(logits, masks) * batch_size

            metrics = calculate_metrics(logits, masks)
            test_precision += metrics['precision'] * batch_size
            test_recall += metrics['recall'] * batch_size
            test_f1 += metrics['f1'] * batch_size

            test_progress.set_postfix({
                'loss': loss.item(),
                'iou': calculate_iou(logits, masks),
                'dice': calculate_dice(logits, masks)
            })

    test_loss /= len(test_dataset)
    test_iou /= len(test_dataset)
    test_dice /= len(test_dataset)
    test_precision /= len(test_dataset)
    test_recall /= len(test_dataset)
    test_f1 /= len(test_dataset)

    print(
        f"测试结果: Loss={test_loss:.4f}, IoU={test_iou:.4f}, Dice={test_dice:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}")

    # 保存测试结果
    with open(os.path.join(OUTPUT_DIR, "test_results.txt"), 'w') as f:
        f.write(f"测试结果:\n")
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"IoU: {test_iou:.4f}\n")
        f.write(f"Dice: {test_dice:.4f}\n")
        f.write(f"Precision: {test_precision:.4f}\n")
        f.write(f"Recall: {test_recall:.4f}\n")
        f.write(f"F1: {test_f1:.4f}\n")


# 运行训练
if __name__ == "__main__":
    # 设置debug=True可以启用调试输出
    train_model(debug=False)
