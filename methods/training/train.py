import datetime
import math
import os
from typing import Union
from functools import partial

import numpy as np
import time
import torch
import torch.nn as nnmic
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
# import timm
# import timm.optim
# import timm.scheduler

from LSD.datasets import FastDataLoader
from finetuneing.config import Config
from finetuneing.models import load_checkpoint, save_checkpoint
from finetuneing.utils import *
import finetuneing.utils.help_builder as help_builder
from .postprocess import process_outputs
from .preprocess import SeismicDataset,SFTDataset
from .validate import validate, visualize_GT
from .schedule import WarmupCosineAnnealingLR
from optim import get_all_parameters

from finetuneing.training.lars import LARS
import LSD.models.backbone_ablation as backbone_ablation

from mup import set_base_shapes

from finetuneing.datasets.SFTData import get_id_type, get_loss, DistributedWeightedSampler

import matplotlib.pyplot as plt
import pandas as pd


def check_data(
    args
):
    if is_main_process():
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
    # Data loader
    model_inputs = [['z', 'n', 'e']]
    model_labels, model_tasks = help_builder.get_labels_tasks(args.downstream_task)
    train_dataset = SFTDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="train",
    )
    print(f"train size: {len(train_dataset)}")

    train_sampler = (
        DistributedWeightedSampler(train_dataset)
        if is_dist_avail_and_initialized()
        else None
    )

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # Save and display metrics
    fname = f'mag_res_{dist.get_rank():0>2}.csv'
    print(f'fname is: {fname}')

    progress = ProgressMeter(
        len(data_loader),
        [],
        prefix=f"Train: [epoch 0/{args.epochs}]",
    )

    for epoch in range(args.start_epoch, args.epochs):
        data = []
        save_path = os.path.join(args.log_dir,f'ep{epoch}') 
        if is_main_process():
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        for step, (_, loss_targets, _) in enumerate(data_loader):
            # visualize_data(x,'/public/home/zhangbei/work_dir/DiTing/visualize',metrics_targets,step)
            data.append(loss_targets.ravel())

            if step % args.log_step == 0 and is_main_process():
                prg_str = progress.get_str(batch_idx=step, name=f"{args.model_name}_train")
                print(prg_str)
        data = torch.cat(data).cpu().numpy()
        res = pd.DataFrame({'mag':data,'rank':np.ones_like(data)})
        res.to_csv(os.path.join(save_path,fname), index=False)

def train(
    args,
    tasks,
    model,
    optimizer,
    scheduler,
    loss_fn,
    train_loader,
    epoch,
    device,
    tensor_writer,
) -> Union[list, dict]:
    model.train()

    # Save and display metrics
    train_loss_per_step = []
    average_meters = {}
    metrics_merged = {}
    sampling_rate = train_loader.dataset.sampling_rate()

    for task in tasks:
        metrics = Metrics(
            task=task,
            metric_names=Config.get_metrics(task),
            sampling_rate=sampling_rate,
            time_threshold=args.time_threshold,
            num_samples=args.in_samples,
            device=device,
        )
        metrics_merged[f"{task}"] = metrics
        for metric in metrics.metric_names():
            average_meters[f"{task}_{metric}"] = AverageMeter(
                f"[{task.upper()}]{metric}", ":6.4f"
            )

    average_meters["loss"] = AverageMeter("Loss", ":6.4f")
    average_meters["Time"] = AverageMeter("Time", ":6.4f")
    average_meters["Data Time"] = AverageMeter("Data Time", ":6.4f")
    if args.splitPS:
        average_meters["Pn_index_loss"] = AverageMeter("Pn_index_loss", ":6.4f")
        average_meters["Pg_index_loss"] = AverageMeter("Pg_index_loss", ":6.4f")
        average_meters["Sn_index_loss"] = AverageMeter("Sn_index_loss", ":6.4f")
        average_meters["Sg_index_loss"] = AverageMeter("Sg_index_loss", ":6.4f")
        average_meters["det_loss"] = AverageMeter("det_loss", ":6.4f")
        

    progress = ProgressMeter(
        len(train_loader),
        [m for m in average_meters.values()],
        prefix=f"Train: [{epoch}/{args.epochs}]",
    )
    """
    (
        label_names,
        tgts_trans_for_loss,
        outs_trans_for_loss,
        outs_trans_for_res,
    ) = Config.get_model_config_(
        args.model_name,
        "labels",
        "targets_transform_for_loss",
        "outputs_transform_for_loss",
        "outputs_transform_for_results",
    )
    """
    label_names, _ = help_builder.get_labels_tasks(args.downstream_task)
    tgts_trans_for_loss = None
    outs_trans_for_loss = None 
    outs_trans_for_res = None
    # outs_trans_for_loss = None
    # outs_trans_for_res = None
    end = time.time()
    iters_per_epoch = len(train_loader)
    for step, (x, loss_targets, metrics_targets, info_for_logging) in enumerate(train_loader):
        # visualize_data(x,'/public/home/zhangbei/work_dir/DiTing/visualize',metrics_targets,step)
        average_meters["Data Time"].update(time.time() - end)
        if isinstance(x, (list, tuple)):
            x = [xi.to(device) for xi in x]
        else:
            x = x.to(device)

        if isinstance(loss_targets, (list, tuple)):
            loss_targets = [yi.to(device) for yi in loss_targets]
        else:
            loss_targets = loss_targets.to(device)

        # Forward
        if 'ppks_type' in metrics_targets.keys() and 'spks_type' in metrics_targets.keys():
            outputs = model(x,metrics_targets['ppks_type'],metrics_targets['spks_type'])
        else:
            outputs = model(x)

        # Loss
        outputs_for_loss = (
            outs_trans_for_loss(outputs) if outs_trans_for_loss is not None else outputs
        )
        loss_targets = (
            tgts_trans_for_loss(loss_targets)
            if tgts_trans_for_loss is not None
            else loss_targets
        )
        if 'ppks_type' in metrics_targets.keys() and 'spks_type' in metrics_targets.keys():
            loss,loss_ = loss_fn(outputs_for_loss, loss_targets,show_loss=True)
            loss_dict = get_loss(loss_,metrics_targets['ppks_type'],metrics_targets['spks_type'])
            for key in loss_dict.keys():
                average_meters[f"{key}_loss"].update(loss_dict[key][0],loss_dict[key][1])
        else:
            losses = loss_fn(outputs_for_loss, loss_targets)
            loss = losses.mean()

        if is_main_process() and args.visualize and step % 100000 == 0 and 'ppk' in label_names[0]:
            # Only applicable to phase-picking task.
            vis_waves_preds_targets(x[0].detach().cpu().numpy(),
                                    outputs[0].detach().cpu().numpy(),
                                    loss_targets[0].detach().cpu().numpy(),
                                    sampling_rate,
                                    args.visualize_save_dir,
                                    step_epoch=(step,epoch))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # for model_key in ['Pn_index','Pg_index','Sn_index','Sg_index','det']:
        #     for param in model.downstream_heads[model_key].parameters():
        #         print(f'{model_key}:',torch.norm(param.grad,p='fro'))
			
        # Adjust learning rate
        if scheduler is not None:
            current_step = iters_per_epoch * epoch + step
            scheduler(current_step)
            lr = optimizer.param_groups[0]['lr']
            '''
            if args.lr_scheduler == "coswarmup":
                scheduler.step(epoch)
                lr =optimizer.param_groups[0]['lr']
            else:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
            '''
        else:
            lr = optimizer.param_groups[0]["lr"]

        # Batch size of the step
        if isinstance(x, (list, tuple)):
            step_batch_size = x[0].size(0)
        else:
            step_batch_size = x.size(0)

        # Reduce
        '''
        if is_dist_avail_and_initialized():
            loss = reduce_tensor(loss, "AVG")
            step_batch_size = torch.tensor(
                step_batch_size, device=device, dtype=torch.int32
            )
            step_batch_size = reduce_tensor(step_batch_size)
            dist.barrier()
            step_batch_size = step_batch_size.item()
        '''

        # Save loss
        average_meters["loss"].update(loss.item(), step_batch_size)
        train_loss_per_step.append(loss.item())

        # Process outputs
        with torch.no_grad():
            outputs_for_metrics = (
                outs_trans_for_res(outputs) if outs_trans_for_res is not None else outputs
            )
            results, _ = process_outputs(args, outputs_for_metrics, label_names, sampling_rate)

#        if step % 100 == 0 and is_main_process():
#            visualize_GT(data=x,
#                         save_path=os.path.join(args.log_dir,f"plots/ep{epoch:0>2}"),
#                         gts=loss_targets,
#                         metrics_targets=metrics_targets,
#                         outputs_for_loss=outputs_for_loss,
#                         results=results,
#                         losses=losses,
#                         ppks=info_for_logging,
#                         step=step,
#                         mode='train')

        # Calculate metrics
        tasks_metrics = {}
        for task in tasks:
            metrics = Metrics(
                task=task,
                metric_names=Config.get_metrics(task),
                sampling_rate=sampling_rate,
                time_threshold=args.time_threshold,
                num_samples=args.in_samples,
                device=device,
            )
            tasks_metrics[task] = metrics
            metrics.compute(
                targets=metrics_targets[task],
                preds=results[task],
            )
            for metric in metrics.metric_names():
                average_meters[f"{task}_{metric}"].update(
                    metrics.get_metric(name=metric), step_batch_size
                )
            metrics_merged[f"{task}"].add(metrics)

        # Tensorboard
        if tensor_writer is not None and is_main_process():
            gstep = epoch * len(train_loader) + step
            tensor_writer.add_scalar("learning-rate/step", lr, gstep)
            tensor_writer.add_scalar("train-loss/step", loss.item(), gstep)
            tensor_writer.add_scalar("train-time/step", time.time()-end, gstep)
            for task in tasks:
                values = tasks_metrics[task].get_all_metrics()
                tensor_writer.add_scalars(f"train.{task}.metrics/step", values, gstep)
        average_meters["Time"].update(time.time() - end)
        end = time.time()

        if step % args.log_step == 0 and is_main_process():
            prg_str = progress.get_str(batch_idx=step, name=f"{args.model_name}_train")
            print(prg_str)

    return train_loss_per_step, metrics_merged


def train_worker(args, device) -> str:

    checkpoint_save_dir = os.path.join(args.log_dir, "checkpoints")
    tb_dir = os.path.join(args.log_dir, "tensorboard")
    args.visualize_save_dir = os.path.join(args.log_dir, "train_visualize")
    
    tensor_writer = SummaryWriter(tb_dir) if args.use_tensorboard else None

    if is_main_process():
        with open(os.path.join(args.log_dir, f"run_tb_{get_time_str()}.sh"), "w") as f:
            f.write(f"tensorboard --logdir '{tb_dir}' --port 8080")
        if not os.path.exists(checkpoint_save_dir):
            os.makedirs(checkpoint_save_dir)
        if not os.path.exists(args.visualize_save_dir) and args.visualize:
            os.makedirs(args.visualize_save_dir)

    # Data loader
    model_inputs = [['z', 'n', 'e']]
    model_labels, model_tasks = help_builder.get_labels_tasks(args.downstream_task)

    # train_dataset = SeismicDataset(
    #     args=args,
    #     input_names=model_inputs,
    #     label_names=model_labels,
    #     task_names=model_tasks,
    #     mode="train",
    # )
    # val_dataset = SeismicDataset(
    #     args=args,
    #     input_names=model_inputs,
    #     label_names=model_labels,
    #     task_names=model_tasks,
    #     mode="val",
    # )
    train_dataset = SFTDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="train",
    )
    val_dataset = SFTDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="val",
    )

    print(f"train size: {len(train_dataset)}, val size:{len(val_dataset)}")

#    train_sampler = (
#        DistributedWeightedSampler(train_dataset)
#        if is_dist_avail_and_initialized()
#        else None
#    )
    train_sampler = (
        torch.utils.data.DistributedSampler(train_dataset)
        if is_dist_avail_and_initialized()
        else None
    )
    val_sampler = (
        torch.utils.data.DistributedSampler(val_dataset)
        if is_dist_avail_and_initialized()
        else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=val_sampler,
    )

    # Epochs & Steps
    if args.steps > 0:
        args.epochs = math.ceil(args.steps / len(train_loader))
    args.steps = args.epochs * len(train_loader)
    print(f"`args.epochs` -> {args.epochs}, `args.steps` -> {args.steps}")

    # Load checkpoint (resume training)
    if args.checkpoint:
        checkpoint = load_checkpoint(
            args.checkpoint,
            device=device,
            dist_mode=args.distributed,
            compile_mode=args.use_torch_compile,
            resume=True,
        )
        if checkpoint is not None and "epoch" in checkpoint:
            args.start_epoch = checkpoint["epoch"] + 1
        print(f"Model loaded: {args.checkpoint}")
    else:
        checkpoint = None

    # Loss
    loss_fn = help_builder.get_loss(args.downstream_task, args.det_weight, args.p_weight, args.s_weight, args.target_distribution_name)
    best_loss = (
        float("inf")
        if (checkpoint is None or "loss" not in checkpoint)
        else checkpoint["loss"]
    )
    loss_fn = loss_fn.to(device)

    # Model (todo) enable mup init
    base_encoder_size_dict = backbone_ablation.get_encoder_size_dict(width=args.base_width, depth=24) # args.encoder_size
    base_model = help_builder.create_model(
        model_name=args.model_name,
        downstream_tasks=args.downstream_task,
        in_samples=args.in_samples,
        encoder_size=base_encoder_size_dict,
        eval_type=args.eval_type,
        pool_type=args.pool_type,
        args=args,
    )
    target_encoder_size_dict = backbone_ablation.get_encoder_size_dict(width=args.target_width, depth=24)
    model = help_builder.create_model(
        model_name=args.model_name,
        downstream_tasks=args.downstream_task,
        in_samples=args.in_samples,
        encoder_size=target_encoder_size_dict,
        eval_type=args.eval_type,
        pool_type=args.pool_type,
        args=args,
    )
    ### muP: set base_shapes
    set_base_shapes(model, base_model) # do_assert=False

    if checkpoint is not None and "model_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_dict"])
        print(f"model.load_state_dict")

    if args.pretrained:
        assert checkpoint is None, "checkpoint and pretrained cannot be used together"
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            print('############# ckpt keys', checkpoint.keys())
            key = 'state_dict'
            state_dict = checkpoint[key]
            # print(f'############# {key} has ', state_dict.keys())
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if args.pretrain_method == "speed":
                    if k.startswith("module.encoder."):
                        # remove prefix
                        state_dict[k[len("module.encoder.") :]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                elif args.pretrain_method == "mae":
                    if args.pool_type == 'decoder':
                        if k.startswith("module.base_encoder."):
                            state_dict['encoder.'+k[len("module.base_encoder.") :]] = state_dict[k]
                        elif k.startswith("module.base_decoder."):
                            state_dict['decoder.'+k[len("module.base_encoder.") :]] = state_dict[k]
                    else:
                        # from deepspeed
                        '''
                        if k.startswith("base_encoder."):
                            # remove prefix
                            state_dict['0.'+k[len("base_encoder.") :]] = state_dict[k] # for MAE:0. is mapped to encoder(in func:help_builder.create_model)
                        '''
                        # Else
                        if k.startswith("module.base_encoder."):
                            # remove prefix
                            state_dict['0.'+k[len("module.base_encoder.") :]] = state_dict[k] # for MAE:0. is mapped to encoder(in func:help_builder.create_model)
                    
                    del state_dict[k]
                elif args.pretrain_method == "byol":
                    if k.startswith("module.base_encoder."):
                        # remove prefix
                        state_dict[k[len("module.base_encoder.") :]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                elif args.pretrain_method == "lp":
                    if args.pretrained.endswith('.pt'):
                        key = "module"
                    else:
                        key = 'model_dict'
                    state_dict = checkpoint[key]
                    del checkpoint["optimizer_dict"]
                else:
                    raise NotImplementedError(f"Unsupported pretrain method:'{args.pretrain_method}'")
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)

            print(msg)
            
            if args.pool_type == 'cls':
                assert msg.missing_keys == ['2.fc.weight', '2.fc.bias'],"load pretrain model fail!"
            elif args.pool_type == 'avg' or args.pool_type == 'attentive':
                missing_keys_except_attentive = [k for k in msg.missing_keys if not k.startswith('1.')]
                assert missing_keys_except_attentive == ['3.fc.weight', '3.fc.bias'],"load pretrain model fail!"
            # else:
            #    raise NotImplementedError
            
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
            assert os.path.isfile(args.pretrained),"no checkpoint found at '{}'".format(args.pretrained)

    model = model.to(device)

    # https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/main_finetune.py#L267
    eff_batch_size = args.batch_size * misc.get_world_size()
    args.lr = args.base_lr # * eff_batch_size / 256
    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)
    
    # Optimizer
    print('===> Constructing criterion and optimizer....')
    optimizer_params = get_all_parameters(args, model)
    optim_lower = args.optim.lower()
    if optim_lower == "adam":
        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif optim_lower == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif optim_lower == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    # add lars
    elif optim_lower == "lars":
        optimizer = LARS(
            params=optimizer_params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer:'{args.optim}'")
    print(args.checkpoint)    
    if args.pretrain_method == "lp":
        del checkpoint["optimizer_dict"]
        print("optimizer_dict")
    if checkpoint is not None and "optimizer_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_dict"])
        print(f"optimizer.load_state_dict")


    current_steps = args.start_epoch * len(train_loader) - 1

    # create scheduler if train
    if args.use_lr_scheduler:
        args.warmup = int(args.steps * args.warmup_steps) # default: 2%
        args.drop_step = int(args.steps * args.down_steps) # default: 10%
        if args.lr_scheduler == 'cosine':
            scheduler = warmup_cosine_lr(optimizer, args, args.steps)
        elif args.lr_scheduler == 'wsd':
            scheduler = warmup_stable_lr(optimizer, args, args.steps)
        else:
            raise NotImplementedError(f'{args.lr_schedule} is not supported, choose from [cosine, wsd]')
    else:
        scheduler = None

    # Save loss
    losses_dict = {
        n: []
        for n in ["train_loss_per_step", "train_loss_per_epoch", "val_loss_per_epoch"]
    }

    num_saved = 0
    epochs_since_improvement = 0
    if is_dist_avail_and_initialized():
        local_rank = get_local_rank()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=args.find_unused_parameters,
        )
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ckpt_path = None
    cost_time = datetime.timedelta()
    for i, epoch in enumerate(range(args.start_epoch, args.epochs)):
        epoch_start_time = datetime.datetime.now()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch=epoch)

        # Train
        train_losses, train_metrics_dict = train(
            args,
            model_tasks,
            model,
            optimizer,
            scheduler,
            loss_fn,
            train_loader,
            epoch,
            device,
            tensor_writer,
        )
        train_loss = np.mean(train_losses)
        losses_dict["train_loss_per_step"].extend(train_losses)
        losses_dict["train_loss_per_epoch"].append(train_loss)

        # Validate
        val_loss, val_metrics_dict = validate(
            args, model_tasks, model, loss_fn, val_loader, epoch, device
        )
        losses_dict["val_loss_per_epoch"].append(val_loss)

        if is_main_process():
            # Save best model
            ckpt_path = os.path.join(checkpoint_save_dir, f"model-latest.pth")
            save_checkpoint(ckpt_path, epoch, model, optimizer, best_loss)
            print(f"Model saved: {ckpt_path}")
            if val_loss < best_loss:
                best_loss = val_loss
                ckpt_path = os.path.join(checkpoint_save_dir, f"model-{epoch}.pth")
                save_checkpoint(ckpt_path, epoch, model, optimizer, best_loss)
                print(f"Model saved: {ckpt_path}")
                num_saved += 1
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                print(f"Epochs since last improvement:{epochs_since_improvement}")

            # Tensorboard
            if tensor_writer is not None:
                tensor_writer.add_scalars(
                    "train-val.loss/epoch",
                    {"train": train_loss, "val": val_loss},
                    epoch,
                )
                for task in model_tasks:
                    tensor_writer.add_scalars(
                        f"train.{task}.metrics/epoch",
                        train_metrics_dict[task].get_all_metrics(),
                        epoch,
                    )
                    tensor_writer.add_scalars(
                        f"val.{task}.metrics/epoch",
                        val_metrics_dict[task].get_all_metrics(),
                        epoch,
                    )
                    tensor_writer.add_scalars(
                        f"val.{task}.allvalues/epoch",
                        val_metrics_dict[task].to_dict(),
                        epoch,
                    )

            # Save log
            train_metrics_str = "* [Train Metrics]"
            val_metrics_str = "* [Val Metrics]"
            for task in model_tasks:
                train_metrics_str += f"[{task.upper()}]{train_metrics_dict[task]} "
                val_metrics_str += f"[{task.upper()}]{val_metrics_dict[task]} "
            print(train_metrics_str)
            print(val_metrics_str)

            # Early stopping
            if epochs_since_improvement > args.patience:
                print(f"\n* Stop training.")
                break

            # Time
            epoch_end_time = datetime.datetime.now()
            epoch_cost_time = epoch_end_time - epoch_start_time
            cost_time += epoch_cost_time
            estimated_end_time = (
                (cost_time / (i + 1)) * 0.1 + epoch_cost_time * 0.9
            ) * (args.epochs - (i + 1)) + epoch_end_time
            print(f"* Epoch cost time: {strftimedelta(epoch_cost_time)}")
            print(
                f"* Estimated end time: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

    return ckpt_path

def find_latest_file(directory):
    if not os.path.isdir(directory):
        raise ValueError("Provided path is not a directory")

    latest_file = None
    latest_time = 0

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            file_mod_time = os.path.getmtime(filepath)
            if file_mod_time > latest_time:
                latest_time = file_mod_time
                latest_file = filepath

    return latest_file


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def warmup_cosine_lr(optimizer, args, steps):
    def _lr_adjuster(step):
        for param_group in optimizer.param_groups:
            base_lr = param_group.get("base_lr", args.lr) # (todo) 分组设立base lr

            if step < args.warmup:
                lr = _warmup_lr(base_lr, args.warmup, step)
            else:
                e = step - args.warmup
                es = steps - args.warmup
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = scale * lr
        return lr
    return _lr_adjuster


# following https://github.com/OpenBMB/MiniCPM/issues/73
def warmup_stable_lr(optimizer, args, steps):
    def _lr_adjuster(step):
        for param_group in optimizer.param_groups:
            base_lr = param_group.get("base_lr", args.lr) # (todo) 分组设立base lr

            if step < args.warmup: # warmup stage
                lr = _warmup_lr(base_lr, args.warmup, step)
            elif step > steps - args.drop_step: # decay stage
                start_decay_step = steps - args.drop_step
                e = step - start_decay_step
                es = args.drop_step
                process = e / es
                lr = 0.5 * (1 + np.cos(np.pi * process)) * base_lr
            else: # stable stage
                lr = base_lr
            
            scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = scale * lr
        return lr
    return _lr_adjuster
