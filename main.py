import sys
import argparse
import datetime
import numpy as np
import time
import torch

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from vision_transformer import vit_base_patch16_224
from datasets import build_continual_dataloader
from engine import train_and_evaluate, evaluate_till_now
import utils
import os
import wandb

import warnings

warnings.filterwarnings("ignore", "Argument interpolation should be of type InterpolationMode instead of int")


def main(args):
    utils.init_distributed_mode(args)
    utils.fix_seed(args.seed)

    device = torch.device(args.device)
    data_loader, class_mask = build_continual_dataloader(args)

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
    )
    original_model.to(device)
    model.to(device)

    if args.freeze:
        # all parameters are frozen for original vit model
        for p in original_model.parameters():
            p.requires_grad = False

        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        attr_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        obj_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, "checkpoint/task{}_checkpoint.pth".format(task_id + 1))
            if os.path.exists(checkpoint_path):
                print("Loading checkpoint from:", checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model"])
            else:
                print("No checkpoint found at:", checkpoint_path)
                return
            _ = evaluate_till_now(
                model,
                original_model,
                data_loader,
                device,
                task_id,
                class_mask,
                acc_matrix,
                attr_acc_matrix,
                obj_acc_matrix,
                args,
            )

        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    criterion1 = torch.nn.CrossEntropyLoss().to(device)
    criterion = [criterion1]

    optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != "constant":
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == "constant":
        lr_scheduler = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(
        model,
        model_without_ddp,
        original_model,
        criterion,
        data_loader,
        optimizer,
        lr_scheduler,
        device,
        class_mask,
        args,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compiler training and evaluation configs")
    config = parser.parse_known_args()[-1][0]

    subparser = parser.add_subparsers(dest="subparser_name")

    if config == "ut_zappos":
        from configs.ut_zappos import get_args_parser

        config_parser = subparser.add_parser("ut_zappos", help="UT-Zappos L2P configs")
    elif config == "clothing16k":
        from configs.clothing16k import get_args_parser

        config_parser = subparser.add_parser("clothing16k", help="Clothing16k L2P configs")
    else:
        raise NotImplementedError(f"{config} not found in configs/")

    get_args_parser(config_parser)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="compiler",
        config=vars(args),
        name=utils.get_run_name(args),
        tags=[
            f"{args.dataset}",
            f"{args.num_tasks}tasks",
        ],
    )

    main(args)
    sys.exit(0)
