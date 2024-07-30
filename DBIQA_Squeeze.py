import torch
from torchvision import models, transforms
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 这个adaptiveAVG是把输入按spatial打成对应得维度
        self.max_pool = nn.AdaptiveMaxPool2d(1) # 作用同上

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) # 先空间平均作为raw权重 然后再在channel位置放大缩小得到自适应权重
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x.mul(self.sigmoid(out))

class HighDimProj(nn.Module):
    def __init__(self, in_planes):
        super(HighDimProj, self).__init__()
        self.fc1 = nn.Conv2d(in_planes, 512, 1, bias=False)
        self.relu1 = nn.ReLU()

    def forward(self, x): # x 的输入格式是：[batch_size, C, H, W]
        out = self.relu1(self.fc1(x))
        return out

def downsample(image, resize=False):
    if resize and min(image.shape[2], image.shape[3]) > 224:
        image = transforms.functional.resize(image,224)
    return image

class Squeeze(torch.nn.Module):
    def __init__(self, requires_grad=False, resize=False):
        super(Squeeze, self).__init__()
        self.chns = [64, 128, 256, 384, 512]
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        SquNet_pretrained_features = models.squeezenet1_1(pretrained=True).features

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        for x in range(0, 2):
            self.stage1.add_module(str(x), SquNet_pretrained_features[x])
        for x in range(2, 5):
            self.stage2.add_module(str(x), SquNet_pretrained_features[x])
        for x in range(5, 8):
            self.stage3.add_module(str(x), SquNet_pretrained_features[x])
        for x in range(8, 11):
            self.stage4.add_module(str(x), SquNet_pretrained_features[x])
        for x in range(11, 13):
            self.stage5.add_module(str(x), SquNet_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def get_features(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1 = h
        h = self.stage2(h)
        h_relu2 = h
        h = self.stage3(h)
        h_relu3 = h
        h = self.stage4(h)
        h_relu4 = h
        h = self.stage5(h)
        h_relu5 = h
        outs = [x, h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]  #

        return outs

    def forward(self, x):
        if self.resize:
            x = downsample(x, resize=True)
        feats_x = self.get_features(x)

        return feats_x

class DBIQA(torch.nn.Module):
    def __init__(self):
        super(DBIQA, self).__init__()
        self.register_parameter('Poly_param', nn.Parameter(torch.tensor([1.0,0.0,1.0])))

        self.AttenA1 = HighDimProj(in_planes=64)
        self.AttenA2 = HighDimProj(in_planes=128)
        self.AttenA3 = HighDimProj(in_planes=256)
        self.AttenA4 = HighDimProj(in_planes=384)
        self.AttenA5 = HighDimProj(in_planes=512)

        self.AttenB1 = ChannelAttention(in_planes=64)
        self.AttenB2 = ChannelAttention(in_planes=128)
        self.AttenB3 = ChannelAttention(in_planes=256)
        self.AttenB4 = ChannelAttention(in_planes=384)
        self.AttenB5 = ChannelAttention(in_planes=512)

    def Kernel(self, tensor, alpha, C, d):
        batchSize, dim, h, w = tensor.data.shape
        M = h * w
        tensor = tensor.reshape(batchSize, dim, M)

        raw   = tensor.bmm(tensor.transpose(1, 2))
        Kernel_matrix =  ( alpha * raw + C )
        Kernel_matrix = torch.clamp(Kernel_matrix, min=1e-17)
        Kernel_matrix = Kernel_matrix.pow(d)

        return Kernel_matrix

    def forward_A(self, x):
        h_relu1 = self.AttenA1(x[1].clone())
        h_relu2 = self.AttenA2(x[2].clone())
        h_relu3 = self.AttenA3(x[3].clone())
        h_relu4 = self.AttenA4(x[4].clone())
        h_relu5 = self.AttenA5(x[5].clone())
        return [x[0], h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

    def forward_B(self, x):
        h_relu1 = self.AttenB1(x[1].clone())
        h_relu2 = self.AttenB2(x[2].clone())
        h_relu3 = self.AttenB3(x[3].clone())
        h_relu4 = self.AttenB4(x[4].clone())
        h_relu5 = self.AttenB5(x[5].clone())
        return [x[0], h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

    def forward_metric(self, x, y, p, q, as_loss):
        b,c,h,w  = p.shape
        Gram_feats0 = self.Kernel(x, alpha=self.Poly_param[0], C=self.Poly_param[1], d=self.Poly_param[2]).reshape(b, -1)
        Gram_feats1 = self.Kernel(y, alpha=self.Poly_param[0], C=self.Poly_param[1], d=self.Poly_param[2]).reshape(b, -1)

        if as_loss==True:
            score1 = torch.abs(Gram_feats0 - Gram_feats1).mean(dim=1)
            score2 = torch.abs(p.reshape(b, c, h * w) - q.reshape(b, c, h * w)).sum(dim=2).mean(dim=1)
        else:
            score1 = torch.abs(Gram_feats0 - Gram_feats1).sum(dim=1)
            score2 = torch.abs(p.reshape(b, c, h * w) - q.reshape(b, c, h * w)).sum(dim=2).mean(dim=1)

        return score1 + score2

    def forward(self, x, y, as_loss=False):
        feat_Ax = self.forward_A(x)
        feat_Ay = self.forward_A(y)
        feat_Bx = self.forward_B(x)
        feat_By = self.forward_B(y)
        score = []
        for i in range(len(feat_Ax)):
            s = self.forward_metric(feat_Ax[i], feat_Ay[i], feat_Bx[i], feat_By[i], as_loss=as_loss)
            score.append(s)

        score = sum(score)
        if as_loss == True:
            return score
        else:
            with torch.no_grad():
                return torch.log(score+1)

if __name__ == '__main__':
    from PIL import Image
    import argparse
    from utils import prepare_image224

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='images/I07.png')
    parser.add_argument('--dist', type=str, default='images/I07_09_04.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref = prepare_image224(Image.open(args.ref).convert("RGB"), resize=True).to(device)
    dist = prepare_image224(Image.open(args.dist).convert("RGB"), resize=True).to(device)

    model = DBIQA().to(device)
    model.load_state_dict(torch.load(
        './weights/nobias5_subMean_Squeeze_PLCC_round2_epoch0.pth',map_location=device))
    model.eval()

    net   = Squeeze().to(device)
    net.eval()

    ref_stage = net(ref)
    dist_stage = net(dist)
    score = model(ref_stage, dist_stage, as_loss=False)
    print('score: %.4f' % score.item())
    # score: 13.5452