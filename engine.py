import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
import wandb

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer

import utils
from SCELoss import SCELoss

# from GCE import GCE
# GCE = GCE()
softmax = nn.Softmax(dim=1)


def train_one_epoch(
    model: torch.nn.Module,
    original_model: torch.nn.Module,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    set_training_mode=True,
    task_id=-1,
    class_mask=None,
    args=None,
):
    model.train(set_training_mode)
    original_model.eval()

    if args.dataset == "czsl-clothing16k":
        dataset = "clothing16k"
        attr_num = 9
        obj_num = 8
        com_num = 35
    elif args.dataset == "czsl-ut-zappos":
        dataset = "ut-zappos"
        attr_num = 15
        obj_num = 12
        com_num = 80

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    device = torch.device(args.device)
    CE = criterion[0]
    GCE = criterion[1]

    C_SCELoss = SCELoss(args.ce, args.rce, com_num)
    A_SCELoss = SCELoss(args.ce, args.rce, attr_num)
    O_SCELoss = SCELoss(args.ce, args.rce, obj_num)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("Lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("Loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    header = f"Train: [Task {task_id+1}] Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]"

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output["pre_logits"]
            else:
                cls_features = None

        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)

        logits = output["com_logits"]
        attr_logits = output["attr_logits"]
        obj_logits = output["obj_logits"]

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            attr_mask, obj_mask = get_primitives_in_this_experience(dataset, mask)
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float("-inf"))
            attr_not_mask = np.setdiff1d(np.arange(attr_num), attr_mask)
            attr_not_mask = torch.tensor(attr_not_mask, dtype=torch.int64).to(device)
            attr_logits = attr_logits.index_fill(dim=1, index=attr_not_mask, value=float("-inf"))
            obj_not_mask = np.setdiff1d(np.arange(obj_num), obj_mask)
            obj_not_mask = torch.tensor(obj_not_mask, dtype=torch.int64).to(device)
            obj_logits = obj_logits.index_fill(dim=1, index=obj_not_mask, value=float("-inf"))

        # attr, obj gt and embddings
        # TODO Use attr_emb and obj_emb
        attribute_gt, object_gt = get_primitives_gt(dataset, target)
        com_loss = C_SCELoss(logits, target)  # base criterion (CrossEntropyLoss)
        attr_loss = A_SCELoss(attr_logits, attribute_gt)
        obj_loss = O_SCELoss(obj_logits, object_gt)
        loss = com_loss + args.alpha * (attr_loss + obj_loss)

        if args.pull_constraint and "reduce_sim" in output:
            loss = loss - args.pull_constraint_coeff * (
                args.pull_constraint_coeff_1 * output["reduce_sim"]
                + (output["obj_reduce_sim"] + output["attr_reduce_sim"])
            )
        loss = loss + args.dll_coeff * output["ddl"] + args.ortho * output["ortho"]
        total_logits = compute_final_score_train(dataset, logits, attr_logits, obj_logits, args.beta)
        acc1, acc5 = accuracy(total_logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            message = "Loss is {}, stopping training".format(loss.item())
            print(message)
            wandb.alert(message)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["Acc@1"].update(acc1.item(), n=input.shape[0])
        metric_logger.meters["Acc@5"].update(acc5.item(), n=input.shape[0])

        wandb.log(
            {
                f"Task-{task_id}/Loss": loss.item(),
                f"Task-{task_id}/Lr": optimizer.param_groups[0]["lr"],
                f"Task-{task_id}/Acc@1": acc1.item(),
                f"Task-{task_id}/Acc@5": acc5.item(),
            }
        )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    original_model: torch.nn.Module,
    data_loader,
    device,
    task_id=-1,
    class_mask=None,
    args=None,
):
    if args.dataset == "czsl-clothing16k":
        dataset = "clothing16k"
        attr_num = 9
        obj_num = 8
        com_num = 35
    elif args.dataset == "czsl-ut-zappos":
        dataset = "ut-zappos"
        attr_num = 15
        obj_num = 12
        com_num = 80

    # CE = torch.nn.CrossEntropyLoss()
    C_SCELoss = SCELoss(args.ce, args.rce, com_num)
    A_SCELoss = SCELoss(args.ce, args.rce, attr_num)
    O_SCELoss = SCELoss(args.ce, args.rce, obj_num)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: [Task {}]".format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output["pre_logits"]
            else:
                cls_features = None

            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output["com_logits"]

            attr_logits = output["attr_logits"]
            attr_logits = softmax(attr_logits)

            obj_logits = output["obj_logits"]
            obj_logits = softmax(obj_logits)

            attribute_gt, object_gt = get_primitives_gt(dataset, target)

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[task_id]
                attr_mask, obj_mask = get_primitives_in_this_experience(dataset, mask)

                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)

                logits = logits.index_fill(dim=1, index=not_mask, value=float("-inf"))
                attr_not_mask = np.setdiff1d(np.arange(attr_num), attr_mask)
                attr_not_mask = torch.tensor(attr_not_mask, dtype=torch.int64).to(device)
                attr_logits = attr_logits.index_fill(dim=1, index=attr_not_mask, value=float("-inf"))

                obj_not_mask = np.setdiff1d(np.arange(obj_num), obj_mask)
                obj_not_mask = torch.tensor(obj_not_mask, dtype=torch.int64).to(device)
                obj_logits = obj_logits.index_fill(dim=1, index=obj_not_mask, value=float("-inf"))

            com_loss = C_SCELoss(logits, target)  # base criterion (CrossEntropyLoss)
            attr_loss = A_SCELoss(attr_logits, attribute_gt)
            obj_loss = O_SCELoss(obj_logits, object_gt)
            loss = com_loss + args.alpha * (attr_loss + obj_loss)

            total_logits = compute_final_score(dataset, logits, attr_logits, obj_logits, args.beta)
            acc1, acc5 = accuracy(total_logits, target, topk=(1, 5))

            attr_acc1, attr_acc5 = accuracy_primitives(total_logits, attribute_gt, dataset, "attr", topk=(1, 5))
            obj_acc1, obj_acc5 = accuracy_primitives(total_logits, object_gt, dataset, "obj", topk=(1, 5))

            metric_logger.meters["Loss"].update(loss.item())
            metric_logger.meters["Acc@1"].update(acc1.item(), n=input.shape[0])
            metric_logger.meters["Acc@5"].update(acc5.item(), n=input.shape[0])

            metric_logger.meters["Attr_Acc@1"].update(attr_acc1.item(), n=input.shape[0])
            metric_logger.meters["Attr_Acc@5"].update(attr_acc5.item(), n=input.shape[0])
            metric_logger.meters["Obj_Acc@1"].update(obj_acc1.item(), n=input.shape[0])
            metric_logger.meters["Obj_Acc@5"].update(obj_acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Attr_Acc@1 {attr_top1.global_avg:.3f} Obj_Acc@1 {obj_top1.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.meters["Acc@1"],
            attr_top1=metric_logger.meters["Attr_Acc@1"],
            obj_top1=metric_logger.meters["Obj_Acc@1"],
            losses=metric_logger.meters["Loss"],
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(
    model: torch.nn.Module,
    original_model: torch.nn.Module,
    data_loader,
    device,
    task_id=-1,
    class_mask=None,
    acc_matrix=None,
    attr_acc_matrix=None,
    obj_acc_matrix=None,
    args=None,
):
    stat_matrix = np.zeros((4, args.num_tasks))

    for i in range(task_id + 1):
        test_stats = evaluate(
            model=model,
            original_model=original_model,
            data_loader=data_loader[i]["val"],
            device=device,
            task_id=i,
            class_mask=class_mask,
            args=args,
        )
        stat_matrix[0, i] = test_stats["Acc@1"]
        stat_matrix[1, i] = test_stats["Attr_Acc@1"]
        stat_matrix[2, i] = test_stats["Obj_Acc@1"]
        stat_matrix[3, i] = test_stats["Loss"]

        acc_matrix[i, task_id] = test_stats["Acc@1"]
        attr_acc_matrix[i, task_id] = test_stats["Attr_Acc@1"]
        obj_acc_matrix[i, task_id] = test_stats["Obj_Acc@1"]

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
    diagonal = np.diag(acc_matrix)
    attr_diagonal = np.diag(attr_acc_matrix)
    obj_diagonal = np.diag(obj_acc_matrix)
    result_str = (
        "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAttr_Acc@1: {:.4f}\tObj_Acc@1: {:.4f}\tLoss: {:.4f}".format(
            task_id + 1, avg_stat[0], avg_stat[1], avg_stat[2], avg_stat[3]
        )
    )
    wandb.log(
        {
            "OVER ALL/Task ID": task_id,
            "OVER ALL/AVG Acc@1": avg_stat[0],
            "OVER ALL/AVG Attr_Acc@1": avg_stat[1],
            "OVER ALL/AVG Obj_Acc@1": avg_stat[2],
            "OVER ALL/AVG Loss": avg_stat[3],
        }
    )
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        attr_forgetting = np.mean((np.max(attr_acc_matrix, axis=1) - attr_acc_matrix[:, task_id])[:task_id])
        attr_backward = np.mean((attr_acc_matrix[:, task_id] - attr_diagonal)[:task_id])

        obj_forgetting = np.mean((np.max(obj_acc_matrix, axis=1) - obj_acc_matrix[:, task_id])[:task_id])
        obj_backward = np.mean((obj_acc_matrix[:, task_id] - obj_diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}\tAttr_Forgetting: {:.4f}\tAttr_Backward: {:.4f}\tObj_Forgetting: {:.4f}\tObj_Backward: {:.4f}".format(
            forgetting, backward, attr_forgetting, attr_backward, obj_forgetting, obj_backward
        )

        wandb.log(
            {
                "OVER ALL/forgetting Task ID": task_id,
                "OVER ALL/forgetting": forgetting,
                "OVER ALL/backward": backward,
                "OVER ALL/attr_forgetting": attr_forgetting,
                "OVER ALL/attr_backward": attr_backward,
                "OVER ALL/obj_forgetting": obj_forgetting,
                "OVER ALL/obj_backward": obj_backward,
            }
        )
    print(result_str)

    return test_stats


def train_and_evaluate(
    model: torch.nn.Module,
    model_without_ddp: torch.nn.Module,
    original_model: torch.nn.Module,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    device: torch.device,
    class_mask=None,
    args=None,
):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    attr_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    obj_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = slice(cur_start, cur_end)
                    prev_idx = slice(prev_start, prev_end)

                    with torch.no_grad():
                        if args.distributed:
                            model.module.prompt.prompt.grad.zero_()
                            model.module.prompt.prompt[cur_idx] = model.module.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]["params"] = model.module.parameters()
                        else:
                            model.prompt.prompt.grad.zero_()
                            model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                            optimizer.param_groups[0]["params"] = model.parameters()

        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.prompt.prompt_key.grad.zero_()
                        model.module.prompt.prompt_key[cur_idx] = model.module.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]["params"] = model.module.parameters()
                    else:
                        model.prompt.prompt_key.grad.zero_()
                        model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]["params"] = model.parameters()

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)

        for epoch in range(args.epochs):
            train_stats = train_one_epoch(
                model=model,
                original_model=original_model,
                criterion=criterion,
                data_loader=data_loader[task_id]["train"],
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_norm=args.clip_grad,
                set_training_mode=True,
                task_id=task_id,
                class_mask=class_mask,
                args=args,
            )

            if lr_scheduler:
                lr_scheduler.step(epoch)

        test_stats = evaluate_till_now(
            model=model,
            original_model=original_model,
            data_loader=data_loader,
            device=device,
            task_id=task_id,
            class_mask=class_mask,
            acc_matrix=acc_matrix,
            attr_acc_matrix=attr_acc_matrix,
            obj_acc_matrix=obj_acc_matrix,
            args=args,
        )
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, "checkpoint")).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, "checkpoint/task{}_checkpoint.pth".format(task_id + 1))
            state_dict = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if args.sched is not None and args.sched != "constant":
                state_dict["lr_scheduler"] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and utils.is_main_process():
            with open(
                os.path.join(
                    args.output_dir, "{}_stats.txt".format(datetime.datetime.now().strftime("log_%Y_%m_%d_%H_%M"))
                ),
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")


def get_primitives_in_this_experience(dataset, mask):
    mapping_dict = get_mapping(dataset)
    attr_in_this_experience = [mapping_dict[key][0] for key in mask]
    obj_in_this_experience = [mapping_dict[key][1] for key in mask]
    attr_in_this_experience = list(set(attr_in_this_experience))
    obj_in_this_experience = list(set(obj_in_this_experience))

    return attr_in_this_experience, obj_in_this_experience


def get_mapping(dataset):
    fpath = "local_datasets/" + dataset + "/com2pri.txt"
    with open(fpath, "r") as f:
        mapping_content = f.read()

    mapping_dict = {}
    lines = mapping_content.split("\n")
    for line in lines:
        parts = line.split(":")
        if len(parts) == 2:
            key = int(parts[0])
            values = list(map(int, parts[1].split(",")))
            mapping_dict[key] = values
    return mapping_dict


def get_primitives_gt(dataset, mb_y):
    if dataset.startswith("czsl-"):
        dataset = dataset.lstrip("czsl-")
    mapping_dict = get_mapping(dataset)
    attr_gt = torch.zeros_like(mb_y)
    obj_gt = torch.zeros_like(mb_y)

    for i, val in enumerate(mb_y):
        mapping = mapping_dict[val.item()]
        attr_gt[i] = mapping[0]
        obj_gt[i] = mapping[1]

    return attr_gt, obj_gt


def compute_final_score_train(dataset, com_logits, attr_logits, obj_logits, beta):
    mapping_dict = get_mapping(dataset)
    attr = torch.tensor([value[0] for value in mapping_dict.values()])
    obj = torch.tensor([value[1] for value in mapping_dict.values()])
    attr_logits_clone = attr_logits.clone()
    obj_logits_clone = obj_logits.clone()
    attr_logit = attr_logits_clone[:, attr]
    obj_logit = obj_logits_clone[:, obj]
    mask = com_logits != float("-inf")
    intermediate = torch.zeros_like(com_logits)
    intermediate[mask] = com_logits[mask] + beta * (attr_logit[mask] + obj_logit[mask])
    result = torch.where(mask, intermediate, com_logits)
    return result


def compute_final_score(dataset, com_logits, attr_logits, obj_logits, beta):
    mapping_dict = get_mapping(dataset)
    attr = torch.tensor([value[0] for value in mapping_dict.values()])
    obj = torch.tensor([value[1] for value in mapping_dict.values()])
    attr_logits_clone = attr_logits.clone()
    obj_logits_clone = obj_logits.clone()
    attr_logit = attr_logits_clone[:, attr]
    obj_logit = obj_logits_clone[:, obj]
    com_logit = com_logits + beta * (attr_logit + obj_logit)
    return com_logit


def get_logits_primitives(dataset, mb_y):
    mapping_dict = get_mapping(dataset)
    attribute_gt = torch.zeros_like(mb_y)
    object_gt = torch.zeros_like(mb_y)

    for i, val in enumerate(mb_y.view(-1)):
        mapping = mapping_dict[val.item()]
        attribute_gt.view(-1)[i] = mapping[0]
        object_gt.view(-1)[i] = mapping[1]

    attribute_gt = attribute_gt.view(mb_y.size())
    object_gt = object_gt.view(mb_y.size())

    return attribute_gt, object_gt


def accuracy_primitives(output, target, datasets, primitives, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    attribute_pre, object_pre = get_logits_primitives(datasets, pred)
    if primitives == "attr":
        pred = attribute_pre.t()
    else:
        pred = object_pre.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]
