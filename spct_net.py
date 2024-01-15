import mindspore.nn as nn
from mindspore.ops import operations as ops
from mindspore import Tensor
from .backbone import ResNeXt3D, ResNeXtBottleneck
from .transformers import TransformerBlock


class conv_3d(nn.Cell):
    """
    3D Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_3d, self).__init__()

        self.conv = nn.SequentialCell(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, has_bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, has_bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )

    def construct(self, x):
        x = self.conv(x)
        return x


class AttentionBlock(nn.Cell):
    def __init__(self, in_channels, skip_channels, mid_channels):
        super(AttentionBlock, self).__init__()
        self.W_skip = nn.SequentialCell(
            nn.Conv3d(skip_channels, mid_channels, kernel_size=1, has_bias=False),
            nn.BatchNorm3d(mid_channels)
        )
        self.W_x = nn.SequentialCell(
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, has_bias=False),
            nn.BatchNorm3d(mid_channels)
        )
        self.psi = nn.SequentialCell(
            nn.Conv3d(mid_channels, 1, kernel_size=1, has_bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def construct(self, x_skip, x):
        x_skip = self.W_skip(x_skip)
        x = self.W_x(x)
        out = self.psi(self.relu(x_skip + x))
        return out


class SPCT(nn.Cell):
    def __init__(self, num_classes=2):
        super(SPCT, self).__init__()
        self.backbone = ResNeXt3D(ResNeXtBottleneck, [3, 4, 6, 3])

        n_ch = 64
        self.down3 = nn.SequentialCell(
            nn.Conv3d(n_ch * 16, n_ch * 2, kernel_size=1, has_bias=False),
            nn.BatchNorm3d(n_ch * 2),
            nn.ReLU()
        )
        self.down2 = nn.SequentialCell(
            nn.Conv3d(n_ch * 8, n_ch * 2, kernel_size=1, has_bias=False),
            nn.BatchNorm3d(n_ch * 2),
            nn.ReLU()
        )
        self.down1 = nn.SequentialCell(
            nn.Conv3d(n_ch * 4, n_ch * 2, kernel_size=1, has_bias=False),
            nn.BatchNorm3d(n_ch * 2),
            nn.ReLU()
        )

        # First Transformer blocks
        self.tran1 = TransformerBlock(in_channels=n_ch * 2, layers=4, mlp_dim=n_ch * 4, patch_size=4)
        self.tran2 = TransformerBlock(in_channels=n_ch * 2, layers=4, mlp_dim=n_ch * 4, patch_size=2)
        self.tran3 = TransformerBlock(in_channels=n_ch * 2, layers=4, mlp_dim=n_ch * 4, patch_size=1)

        self.fuse = nn.SequentialCell(
            nn.Conv3d(n_ch * 6, 2 * n_ch, kernel_size=1, has_bias=False),
            nn.BatchNorm3d(2 * n_ch),
            nn.ReLU(),
            nn.Conv3d(2 * n_ch,  2 * n_ch, kernel_size=3, padding=1, has_bias=False),
            nn.BatchNorm3d(2 * n_ch),
            nn.ReLU()
        )

        # voxel attention
        self.attention3 = AttentionBlock(n_ch * 2, n_ch * 2, n_ch)
        self.attention2 = AttentionBlock(n_ch * 2, n_ch * 2, n_ch)
        self.attention1 = AttentionBlock(n_ch * 2, n_ch * 2, n_ch)

        self.refine3 = nn.SequentialCell(
            nn.Conv3d(n_ch * 4,  2 *n_ch, kernel_size=1, has_bias=False),
            nn.Conv3d(2 * n_ch, 2 * n_ch, kernel_size=3, groups=2 * n_ch, padding=1, has_bias=False),
            nn.Conv3d(2 * n_ch, 2 * n_ch, kernel_size=3, groups=2 * n_ch, padding=1, has_bias=False),
            nn.BatchNorm3d(2 * n_ch),
            nn.ReLU(),
            TransformerBlock(in_channels=2 * n_ch, layers=6, mlp_dim=n_ch * 4, patch_size=4)
        )
        self.refine2 = nn.SequentialCell(
            nn.Conv3d(n_ch * 4,  2 * n_ch, kernel_size=1, has_bias=False),
            nn.Conv3d(2 * n_ch, 2 * n_ch, kernel_size=3, groups=2 * n_ch, padding=1, has_bias=False),
            nn.Conv3d(2 * n_ch, 2 * n_ch, kernel_size=3, groups=2 * n_ch, padding=1, has_bias=False),
            nn.BatchNorm3d(2 * n_ch),
            nn.ReLU(),
            TransformerBlock(in_channels=2 * n_ch, layers=6, mlp_dim=n_ch * 4, patch_size=4)
        )
        self.refine1 = nn.SequentialCell(
            nn.Conv3d(n_ch * 4,  2 * n_ch, kernel_size=1, has_bias=False),
            nn.Conv3d(2 * n_ch, 2 * n_ch, kernel_size=3, groups=2 * n_ch, padding=1, has_bias=False),
            nn.Conv3d(2 * n_ch, 2 * n_ch, kernel_size=3, groups=2 * n_ch, padding=1, has_bias=False),
            nn.BatchNorm3d(2 * n_ch),
            nn.ReLU(),
            TransformerBlock(in_channels=2 * n_ch, layers=6, mlp_dim=n_ch * 4, patch_size=4)
        )

        self.last_refine = nn.SequentialCell(
            nn.Conv3d(384, 128, kernel_size=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            conv_3d(128, 128)
        )

        self.predict3 = nn.Conv3d(128, num_classes, kernel_size=1)
        self.predict2 = nn.Conv3d(128, num_classes, kernel_size=1)
        self.predict1 = nn.Conv3d(128, num_classes, kernel_size=1)

        self.predict = nn.Conv3d(128, num_classes, kernel_size=1)

    def construct(self, x):
        features = self.backbone(x)
        layer1, layer2, layer3 = features[0], features[1], features[2]

        # FPN, Top-down
        down3 = self.down3(layer3)
        down2 = ops.Add()(
            ops.ResizeBilinear()(down3, (layer2.shape[2], layer2.shape[3], layer2.shape[4])),
            self.down2(layer2)
        )
        down1 = ops.Add()(
            ops.ResizeBilinear()(down2, (layer1.shape[2], layer1.shape[3], layer1.shape[4])),
            self.down1(layer1)
        )

        # 上采样到相同尺寸
        down3 = ops.ResizeBilinear()(self.tran3(down3), (layer1.shape[2], layer1.shape[3], layer1.shape[4]))
        down2 = ops.ResizeBilinear()(self.tran2(down2), (layer1.shape[2], layer1.shape[3], layer1.shape[4]))
        down1 = self.tran1(down1)

        # 特征融合以及像素注意力
        fuse = self.fuse(ops.Concat(1)((down3, down2, down1)))

        attention3 = self.attention3(fuse, down3)
        attention2 = self.attention2(fuse, down2)
        attention1 = self.attention1(fuse, down1)

        # 把上一步获得的 attention maps 应用到不同尺度的 features maps 上
        refine3 = self.refine3(ops.Concat(1)((down3, attention3 * fuse)))
        refine2 = self.refine2(ops.Concat(1)((down2, attention2 * fuse)))
        refine1 = self.refine1(ops.Concat(1)((down1, attention1 * fuse)))

        final_refine = self.last_refine(ops.Concat(1)((refine1, refine2, refine3)))

        predict3 = self.predict3(refine3)
        predict2 = self.predict2(refine2)
        predict1 = self.predict1(refine1)

        predict = self.predict(final_refine)

        predict1 = ops.ResizeBilinear()(predict1, (x.shape[2], x.shape[3], x.shape[4]))
        predict2 = ops.ResizeBilinear()(predict2, (x.shape[2], x.shape[3], x.shape[4]))
        predict3 = ops.ResizeBilinear()(predict3, (x.shape[2], x.shape[3], x.shape[4]))

        predict = ops.ResizeBilinear()(predict, (x.shape[2], x.shape[3], x.shape[4]))

        if self.training:
            return predict1, predict2, predict3, predict
        else:
            return predict

