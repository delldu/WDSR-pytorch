import random
import PIL.Image as pil_image
import numpy as np
import torch

def load_img(path):
    return pil_image.open(path).convert('RGB')


def img2np(x):
    return np.array(x)


def np2tensor(x):
    return torch.from_numpy(x).permute(2, 0, 1).float()

# xxxx3333
# def tensor2img(x):
#     return pil_image.fromarray(x.byte().cpu().numpy())


def quantize(x, quantize_range):
    return x.clamp(quantize_range[0], quantize_range[1])


def normalize(x, max_value):
    return x * (max_value / 255.0)


def augment_patches(patches, hflip=True, vflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    ret = []
    for p in patches:
        if hflip:
            p = np.fliplr(p).copy()
        if vflip:
            p = np.flipud(p).copy()
        if rot90:
            p = np.rot90(p, axes=(1, 0)).copy()  # Clockwise
        ret.append(p)
    return ret


# xxxx1111
def get_patch(lr, hr, patch_size, scale, augment_patch=False):
    lr_h, lr_w = lr.shape[:2]
    lr_p = patch_size // scale
    lr_x, lr_y = random.randrange(0, lr_w - lr_p + 1), random.randrange(0, lr_h - lr_p + 1)
    hr_x, hr_y, hr_p = lr_x * scale, lr_y * scale, patch_size
    lr = lr[lr_y:lr_y + lr_p, lr_x:lr_x + lr_p]
    hr = hr[hr_y:hr_y + hr_p, hr_x:hr_x + hr_p]
    if augment_patch:
        lr, hr = augment_patches([lr, hr])
    return lr, hr


# if __name__ == '__main__':
#     import pdb

#     img = pil_image.open("/tmp/a.png")

    # pdb.set_trace()
    # a1 = img2np(img)
    # (Pdb) a1.dtype, a1.shape,a1.max(), a1.min()
    # (dtype('uint8'), (540, 960, 3), 203, 0)

    # (Pdb) t1 = np2tensor(a1)
    # (Pdb) t1.dtype, t1.size(), t1.max(), t1.min()
    # (torch.float32, torch.Size([3, 540, 960]), tensor(203.), tensor(0.))
