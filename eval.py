from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from core.option import parser
from core.model import WDSR_A, WDSR_B
# from core.data.div2k import DIV2K
from core.data.sdr import SRDataset

from core.data.utils import quantize
from core.utils import AverageMeter, calc_psnr, load_checkpoint, load_weights
import torchvision.transforms

import pdb

def forward(x):
    with torch.no_grad():
        sr = model(x)
        return sr


def forward_x8(x):
    x = x.squeeze(0).permute(1, 2, 0)

    with torch.no_grad():
        sr = []

        for rot in range(0, 4):
            for flip in [False, True]:
                _x = x.flip([1]) if flip else x
                _x = _x.rot90(rot)
                out = model(_x.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
                out = out.rot90(4 - rot)
                out = out.flip([1]) if flip else out
                sr.append(out)

        return torch.stack(sr).mean(0).permute(2, 0, 1)


def test(dataset, loader, model, device, args, tag=''):
    psnr = AverageMeter()

    # Set the model to evaluation mode
    model.eval()
    count = 0
    with tqdm(total=len(dataset)) as t:
        t.set_description(tag)

        for data in loader:
            count = count + 1

            lr, hr = data
            # pdb.set_trace()

            lr = lr.to(device)
            hr = hr.to(device)

            if args.self_ensemble:
                sr = forward_x8(lr)
            else:
                sr = forward(lr)

            # pdb.set_trace()
            # (Pdb) lr.size()
            # torch.Size([1, 3, 678, 1020])
            # (Pdb) sr.size()
            # torch.Size([1, 3, 1356, 2040])


            # Quantize results
            sr = quantize(sr, args.rgb_range)

            # Update PSNR
            psnr.update(calc_psnr(sr, hr, scale=args.scale, max_value=args.rgb_range[1]), lr.shape[0])

            t.update(lr.shape[0])

            # if count > 100:
            #     srimg = torchvision.transforms.ToPILImage()(sr.squeeze().div(255).cpu())
            #     srimg.save("result/{:03d}.png".format(count - 100))

            #     #     srimg.show()
            #     #     pdb.set_trace()


    print('SDR (val) PSNR: {:.4f} dB'.format(psnr.avg))


if __name__ == '__main__':
    # Define specific options and parse arguments
    parser.add_argument('--dataset-dir', type=str, required=True, help='DIV2K Dataset Root Directory')
    parser.add_argument('--checkpoint-file', type=str, required=True)
    parser.add_argument('--self-ensemble', action='store_true')
    args = parser.parse_args()

    # Set cuDNN auto-tuner and get device
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create model
    if args.model == 'WDSR-B':
        model = WDSR_B(args).to(device)
    else:
        model = WDSR_A(args).to(device)

    # Load weights
    model = load_weights(model, load_checkpoint(args.checkpoint_file)['state_dict'])

    # Prepare dataset
    dataset = SRDataset("test")
    dataloader = DataLoader(dataset=dataset, batch_size=1)

    # pdb.set_trace()

    test(dataset, dataloader, model, device, args)
