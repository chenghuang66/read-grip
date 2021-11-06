import torch
import torch.nn as nn
# 空间图
class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        # 参数kernel_size，stride, padding，dilation也可以是一个int的数据，此时卷积height和width值相同;
        # 也可以是一个tuple数组，tuple的第一维度表示height的数值，tuple的第二维度表示width的数值
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        # print('A.shape ',A.shape)
        assert A.size(1) == self.kernel_size
        # print('1',x.size())
        x = self.conv(x)
        # print('2',x.size())
        n, kc, t, v = x.size()
        # 向下整除//  tensor。view 数组重新排列
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        # print('3',x.size())
        # 爱因斯坦简记法
        x = torch.einsum('nkctv,nkvw->nctw', (x, A))
        # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一样
        return x.contiguous(), A
