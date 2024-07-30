from torchvision import transforms
from torchvision.transforms.functional import to_tensor

def prepare_image224(image, resize = False, repeatNum = 1):
    if resize and min(image.size)>224:
        image = transforms.functional.resize(image,224)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)


