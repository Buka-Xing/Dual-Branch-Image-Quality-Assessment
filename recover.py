import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import imageio

from DBIQA_VGG import DBIQA, VGG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'DBIQA_VGG'
ref_path   = 'images/land3.jpg'
pred_path  = 'images/white.jpg'
save_path  = './results/Enhancement/white_init/%s/iter%dstepsize%.6f.jpg'
final_save_path = './results/Enhancement/white_init/%s/final_result.jpg'
lr = 1e-1
iter_num = 10000
decay = 1000
output_iter = 100

model = DBIQA().to(device)
model.load_state_dict(torch.load(
    './weights/nobias5_subMean_VGG_PLCC_round4_epoch1.pth', map_location=device))
model = model.eval()

net = VGG().to(device)
net.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

ref_img = Image.open(ref_path).convert("RGB")
ref = transform(ref_img).unsqueeze(0)
ref = Variable(ref.float().to(device), requires_grad=False)

Initial  = Image.open(pred_path).convert("RGB")

pred_img = Image.open(pred_path).convert("RGB")
pred = transform(pred_img).unsqueeze(0)
pred = Variable(pred.float().to(device), requires_grad=True)

model.eval()
optimizer = torch.optim.Adam([pred], lr=lr)

for i in range(iter_num + 1):
    pred_stage = net(pred)
    ref_stage  = net(ref)
    dist = model(pred_stage, ref_stage, as_loss=True)
    optimizer.zero_grad()
    dist.backward()
    optimizer.step()
    pred.data.clamp_(min=0,max=1)
    
    if i % output_iter == 0:
        pred_img = pred.squeeze().data.cpu().numpy().transpose(1, 2, 0)
        ref_img2 = ref.squeeze().data.cpu().numpy().transpose(1, 2, 0)

        fig = plt.figure(figsize=(4, 1.5), dpi=300)
        plt.subplot(131)
        plt.imshow(Initial)
        plt.title('initial', fontsize=6)
        plt.axis('off')
        plt.subplot(133)

        plt.imshow(ref_img2)
        plt.title('reference', fontsize=6)
        plt.axis('off')

        plt.subplot(132)       
        plt.imshow(np.clip(pred_img, 0, 1))

        plt.title('iter: %d, dists: %.3g' % ( i, dist.item() ),fontsize=6)
        plt.axis('off')
        plt.savefig(save_path % (model_name,i,lr))
        plt.pause(1)
        plt.cla()
        plt.close()

    if (i+1) % decay == 0:
        lr = max(1e-5, lr * 0.5)
        optimizer = torch.optim.Adam([pred], lr=lr)

imageio.imwrite(final_save_path % (model_name), pred_img)