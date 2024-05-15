# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
if os.environ["W_QUANT"]=='1':
    from pytorch_nndct.apis import torch_quantizer

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger

from tqdm import tqdm


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("--quant_mode", default='calib', type=str, help="mode for quantization")
    parser.add_argument("--quant_dir", default='quantized', type=str, help="directory for quantization")
    parser.add_argument("--is_dump", default=False, action="store_true", help="flag to dump xmodel")
    parser.add_argument("--fast_finetune", default=False, action="store_true", help="fast finetune for quantization")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def feed_model_with_data(model, dataset, device, subset_len, batch_size):
    import random
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if subset_len:
        assert subset_len <= len(dataset)
        subset = torch.utils.data.Subset(dataset, random.sample(range(0, len(dataset)), subset_len))
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, drop_last=False, shuffle=True)
    model.to(device)
    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(tqdm(dataloader)):  
        imgs = imgs.type(torch.cuda.FloatTensor)
        imgs = imgs.to(device)
        outputs = model(imgs)


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = False

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    device = torch.device('cuda')
    if args.is_dump:
        device = torch.device("cpu")
        args.batch_size = 1

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy, quantized=True)

    # torch.cuda.set_device(rank)
    # model.cuda(rank)
    model.to(device)
    model.eval()

    if not args.speed:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint from {}".format(ckpt_file))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_file, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    import copy
    float_model = copy.deepcopy(model)
    if os.environ["W_QUANT"]=='1':
        dummy_input = torch.randn([1, 3, 640, 640]).to(device)
        quantizer = torch_quantizer(args.quant_mode, model, dummy_input, output_dir=args.quant_dir, device=device)
        model = quantizer.quant_model
        model.eval()

    if args.fast_finetune:
        if args.quant_mode == 'calib':
            sample_num = 2000
            fft_batch_size = 50
            quantizer.fast_finetune(feed_model_with_data, (model, evaluator.dataloader.dataset, device, sample_num, fft_batch_size))
        elif args.quant_mode == 'test':
            quantizer.load_ft_param()

    trt_file = None
    decoder = None

    # start evaluate
    *_, summary = evaluator.evaluate(model, float_model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, 
                                     args.is_dump, device)
    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
        exit()
    elif args.quant_mode == 'test' and args.is_dump:
        quantizer.export_xmodel(output_dir=args.quant_dir, deploy_check=False)
        print("****************Completed dumping the xmodel***********************")
        exit()

    logger.info("\n" + summary)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
