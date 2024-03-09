import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.DenUnet import DenUnet
import configs.DenUnet_configs as configs
from trainer import trainer


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/Toothdataset', help='root dir for data')
parser.add_argument('--test_path', type=str,
                    default='./data/Toothdataset/test', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Toothdataset', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=33, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=501, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=10, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--num_workers', type=int,  default=2,
                    help='number of workers')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--output_dir', type=str,
                    default='./results', help='root dir for output log')
parser.add_argument('--model_name', type=str,
                    default='DenUnet')
parser.add_argument('--eval_interval', type=int,
                    default=20, help='evaluation epoch')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')

args = parser.parse_args()

args.output_dir = args.output_dir + f'/{args.model_name}'
os.makedirs(args.output_dir, exist_ok=True)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    CONFIGS = {
        'DenUnet': configs.get_DenUnet_configs(),
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24


    model = DenUnet(config=CONFIGS[args.model_name], img_size=args.img_size, n_classes=args.num_classes).cuda()
    trainer(args, model, args.output_dir)
