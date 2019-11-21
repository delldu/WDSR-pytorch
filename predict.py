#! /usr/bin/env python
# coding=utf-8
#
# /************************************************************************************
# ***
# ***	File Author: Dell, Sat Nov  2 15:26:20 CST 2019
# ***
# ************************************************************************************/
#

"""
Most important !!!
    DO NOT use read_video, write_vido of torchvison for bad quality !!!

    Please use imageio instead of torchvision.io.video !!!
    # https://imageio.readthedocs.io/en/latest/examples.html

    pip install imageio_ffmpeg
"""
import os
import torch
import numpy as np
from core.option import parser
from core.model import WDSR_A, WDSR_B
from core.utils import load_checkpoint, load_weights
from torchvision.io.video import read_video
from PIL import Image
# from torchvision import transforms as transforms
# import argparse
from tqdm import tqdm
# import pdb


def tensor_to_image(t):
    """Convert tensor t to PIL image, torch vision does not good enough."""
    t = t.float().add(0.5).clamp(0, 255).byte().cpu()
    npimg = np.transpose(t.numpy(), (1, 2, 0))
    return Image.fromarray(npimg)


def predict_sr(model, device, input_video, output_dir):
    """Predict SR model."""
    vframes, aframes, info = read_video(input_video, pts_unit='sec')
    # vframe format: [T, H, W, C], data range:[0,255], good for h5!

    for i in tqdm(range(len(vframes))):
        input_tensor = vframes[i].permute(2, 0, 1).float()
        # input_tensor is tensor, format CxHxW, data range [0.0, 1.0]
        input_tensor.unsqueeze_(0)
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_tensor.squeeze_(0)
        output_image = tensor_to_image(output_tensor.cpu())
        output_image.save("{}/{:03d}.png".format(output_dir, i))


if __name__ == '__main__':
    # Define specific options and parse arguments
    parser.add_argument("-i",
                        "--input",
                        help="input video",
                        required=True,
                        type=str)
    parser.add_argument("-o",
                        "--output",
                        help="output video directory, defaut: /tmp/output",
                        type=str,
                        default="/tmp/output")
    parser.add_argument('--checkpoint-file',
                        type=str,
                        default="output/WDSR-B-f32-b16-r6-x4-best.pth.tar")
    args = parser.parse_args()

    # Set cuDNN auto-tuner and get device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create model
    if args.model == 'WDSR-B':
        model = WDSR_B(args).to(device)
    else:
        model = WDSR_A(args).to(device)

    # Load weights
    model = load_weights(model, load_checkpoint(args.checkpoint_file)['state_dict'])

    model.eval()

    # Start SR !
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    predict_sr(model, device, args.input, args.output)
