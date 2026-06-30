#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'sft'))

import argparse
import torch
import numpy as np
import re
from typing import Dict, Any, List

from finetuneing.config import Config
from finetuneing.utils import *
from finetuneing.training.preprocess import SFTDataset
from finetuneing.training.postprocess import process_outputs
from finetuneing.utils.help_builder import get_labels_tasks, get_loss
from finetuneing.datasets.SFTData import get_loss as get_split_loss

from api_client import send_batch_request


def parse_hps(hps: str):
    result = {
        'target_width': 256,
        'base_width': 256,
        'input_mult': 1.0,
        'attn_mult': 256.0,
        'output_mult': 1.0,
        'init_std': 0.02,
        'lr': 0.001,
        'wd': 0.0005,
        'mr': 0.9,
    }
    if not hps:
        return result
    
    # modelsize
    match = re.search(r'modelsize(\d+)', hps)
    if match:
        result['target_width'] = int(match.group(1))
    
    # input-mult
    match = re.search(r'input-mult([\d.]+)', hps)
    if match:
        result['input_mult'] = float(match.group(1))
    
    # attn-mult
    match = re.search(r'attn-mult([\d.]+)', hps)
    if match:
        result['attn_mult'] = float(match.group(1))
    
    # output-mult
    match = re.search(r'output-mult([\d.]+)', hps)
    if match:
        result['output_mult'] = float(match.group(1))
    
    # init-std
    match = re.search(r'init-std([\d.]+)', hps)
    if match:
        result['init_std'] = float(match.group(1))
    
    # lr
    match = re.search(r'lr([\d.]+)', hps)
    if match:
        result['lr'] = float(match.group(1))
    
    # wd
    match = re.search(r'wd([\d.]+)', hps)
    if match:
        result['wd'] = float(match.group(1))
    
    # mr
    match = re.search(r'mr([\d.]+)', hps)
    if match:
        result['mr'] = float(match.group(1))
    
    return result


def create_args():
    args = argparse.Namespace()
    args.checkpoint = "/public/home/zhangbei/work_dir/kangkai/proj_han/output/new/6976/checkpoints/model-latest.pth"
    args.hps = "scaling_mae_mup_ablation/transfered/HPs_schedulecosine_ep100_data400000_modelsize512_lr0.001_wd0.0005_mr0.9_init-std0.08_attn-mult32.0_input-mult14.142135623730951_output-mult2.0"
    hps_parsed = parse_hps(args.hps)
    args.target_width = hps_parsed['target_width']      # 512
    args.base_width = 256
    args.input_mult = hps_parsed['input_mult']          # 14.142135623730951
    args.attn_mult = hps_parsed['attn_mult']            # 32.0
    args.output_mult = hps_parsed['output_mult']        # 2.0
    args.init_std = hps_parsed['init_std']              # 0.08
    args.model_name = "llama"
    args.downstream_task = "cls"
    args.eval_type = "finetune"
    args.pool_type = "avg"
    args.encoder_size = "proxy"
    args.in_samples = 10000
    args.seed = 0
    args.mode = "test"
    args.log_dir = "/public/home/zhangbei/work_dir/kangkai/proj_han/output/new/10932/5e-1"
    args.device = "cuda:0"
    args.pretrained = ""
    args.splitPS = False
    args.patch_size = 50
    args.head_drop_rate = 0.5
    args.drop_path = 0.3
    args.norm_layer = "rmsnorm"
    args.xattn = False
    args.use_torch_compile = False
    
    args.det_weight = 0.1
    args.p_weight = 10.0
    args.s_weight = 10.0
    
    args.batch_size = 50
    args.data_split = True
    args.train_size = 0.9
    args.val_size = 0.1
    args.shuffle = False
    args.workers = 1
    args.pin_memory = True
    
    args.min_snr = -float('inf')
    args.coda_ratio = 2.0
    args.norm_mode = 'std'
    args.p_position_ratio = 0.02
    args.p_position_ratio_type = 'uniform'
    args.p_position_ratio_range_or_sigma = 0.02
    args.band_filt = False
    args.augmentation = False      
    args.label_shape = 'gaussian'
    args.label_width = 0.5
    
    args.add_event_rate = 0.0
    args.add_noise_rate = 0.0
    args.add_gap_rate = 0.0
    args.drop_channel_rate = 0.0
    args.scale_amplitude_rate = 0.0
    args.pre_emphasis_rate = 0.0
    args.pre_emphasis_ratio = 0.97
    args.max_event_num = 1
    args.generate_noise_rate = 0.0
    args.shift_event_rate = 0.0
    args.rotate_event_rate = 0.0
    args.mask_percent = 0
    args.noise_percent = 0
    args.min_event_gap = 0.5
    args.no_event_p = 0.2
    args.random_crop_p = 1.0
    args.weighted_sampling = False
    args.weighted_sampling_key = 'Mag_value'
    
    args.time_threshold = 0.5
    args.min_peak_dist = 1.0
    args.ppk_threshold = 0.3
    args.spk_threshold = 0.3
    args.det_threshold = 0.5
    args.max_detect_event_num = 1
    
    args.subset_names = "cls_data"
    args.train_meta_data_path = "D:\\seisbench_data\\program\\data\\diting\\splited_data\\lp_vt_train_1k.csv"
    args.test_meta_data_path = "D:\\seisbench_data\\program\\data\\diting\\splited_data\\lp_vt_test_1k.csv"
    args.train_data_dir = "D:\\seisbench_data\\program\\data\\diting\\lp_vt.hdf5"
    args.test_data_dir = "D:\\seisbench_data\\program\\data\\diting\\lp_vt.hdf5"
    args.train_sample_num = 10931
    
    args.visualize = False
    args.visualize_save_dir = ""
    args.log_step = 10
    
    print("=" * 60)
    print("📋 Configuration Summary:")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   target_width: {args.target_width}")
    print(f"   input_mult: {args.input_mult}")
    print(f"   attn_mult: {args.attn_mult}")
    print(f"   output_mult: {args.output_mult}")
    print(f"   init_std: {args.init_std}")
    print(f"   train_sample_num: {args.train_sample_num}")
    print("=" * 60)
    
    return args


def run_inference(args, device, server_url="http://127.0.0.1:10089/cls"):
    print("=" * 60)
    print("🚀 Starting Full Inference Pipeline")
    print(f"   Server: {server_url}")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Batch size: {args.batch_size}")
    print("=" * 60)
    
    # ========== 1. preprocess ==========
    print("\n📂 Step 1: Loading and preprocessing data...")
    model_inputs = [['z', 'n', 'e']]
    model_labels, model_tasks = get_labels_tasks(args.downstream_task)

    test_dataset = SFTDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="test",
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
        num_workers=0,
    )

    sampling_rate = test_loader.dataset.sampling_rate()
    total_batches = len(test_loader)
    total_samples = len(test_dataset)
    print(f"   Total samples: {total_samples}")
    print(f"   Total batches: {total_batches}")
    print(f"   Sampling rate: {sampling_rate} Hz")

    # ========== 2. loss ==========
    loss_fn = get_loss(args.downstream_task, args.det_weight, args.p_weight, args.s_weight)
    loss_fn = loss_fn.to(device)

    # ========== 3. init metrics ==========
    average_meters = {}
    metrics_merged = {}

    for task in model_tasks:
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

    progress = ProgressMeter(
        total_batches,
        [m for m in average_meters.values()],
        prefix=f"{'Test'}:",
    )

    # ========== 4. model param ==========
    model_config = {
        'model_name': args.model_name,
        'downstream_task': args.downstream_task,
        'eval_type': args.eval_type,
        'pool_type': args.pool_type,
        'in_samples': args.in_samples,
        'encoder_size': args.encoder_size,
        'base_width': args.base_width,
        'target_width': args.target_width,
        'pretrained': args.pretrained,
        'hps': args.hps,
        'splitPS': args.splitPS,
        'patch_size': args.patch_size,
        'head_drop_rate': args.head_drop_rate,
        'drop_path': args.drop_path,
        'norm_layer': args.norm_layer,
        'xattn': args.xattn,
        'use_torch_compile': args.use_torch_compile,
        'init_std': args.init_std,
        'attn_mult': args.attn_mult,
        'input_mult': args.input_mult,
        'output_mult': args.output_mult,
        'det_weight': args.det_weight,
        'p_weight': args.p_weight,
        's_weight': args.s_weight,
        'device': args.device,
    }

    # ========== 5. batch loop ==========
    print("\n📤 Step 2: Sending batches to API server...")
    failed_batches = []
    success_batches = 0

    with torch.no_grad():
        for step, (x, loss_targets, metrics_targets, info_for_logging) in enumerate(test_loader):
            # print batch shape
            print(f"\n{'='*50}")
            print(f"Batch {step + 1}/{total_batches}")
            print(f"  x.shape: {x.shape}")
            if isinstance(loss_targets, torch.Tensor):
                print(f"  loss_targets.shape: {loss_targets.shape}")
            print(f"{'='*50}")

            x = x.to(device)
            if isinstance(loss_targets, (list, tuple)):
                loss_targets = [yi.to(device) for yi in loss_targets]
            else:
                loss_targets = loss_targets.to(device)

            # API inference
            x_list = x.cpu().numpy().tolist()
            
            result = send_batch_request(
                data=x_list,
                checkpoint=args.checkpoint,
                server_url=server_url,
                timeout=600,
                **model_config
            )

            if result is None:
                failed_batches.append(step)
                continue

            outputs = torch.tensor(result["outputs"], dtype=torch.float32).to(device)
            success_batches += 1

            # loss calculate
            step_batch_size = x.size(0)

            if 'ppks_type' in metrics_targets.keys() and 'spks_type' in metrics_targets.keys():
                loss, loss_ = loss_fn(outputs, loss_targets, show_loss=True)
                loss_dict = get_split_loss(loss_, metrics_targets['ppks_type'], metrics_targets['spks_type'])
                for key in loss_dict.keys():
                    average_meters[f"{key}_loss"].update(loss_dict[key][0], loss_dict[key][1])
                loss = loss.mean()
            else:
                losses = loss_fn(outputs, loss_targets)
                loss = losses.mean()

            average_meters["loss"].update(loss.item(), step_batch_size)

            # postprocess
            results, lls = process_outputs(args, outputs, model_labels, sampling_rate)

            # get metrics
            for task in model_tasks:
                metrics = Metrics(
                    task=task,
                    metric_names=Config.get_metrics(task),
                    sampling_rate=sampling_rate,
                    time_threshold=args.time_threshold,
                    num_samples=args.in_samples,
                    device=device,
                )
                metrics.compute(
                    targets=metrics_targets[task],
                    preds=results[task],
                )
                for metric in metrics.metric_names():
                    average_meters[f"{task}_{metric}"].update(
                        metrics.get_metric(name=metric), step_batch_size
                    )
                metrics_merged[f"{task}"].add(metrics)

            # print step
            if step % args.log_step == 0:
                prg_str = progress.get_str(batch_idx=step, name=f"{args.model_name}_{'test'}")
                print(prg_str)

    # ========== 6. final result ==========
    print("\n" + "=" * 60)
    print("📊 Step 3: Final Results")
    print("=" * 60)
    
    print(f"   Total batches: {total_batches}")
    print(f"   Successful: {success_batches}")
    print(f"   Failed: {len(failed_batches)}")
    if failed_batches:
        print(f"   Failed batch indices: {failed_batches}")

    loss_avg = average_meters["loss"].avg

    test_metrics_str = "* "
    for task in model_tasks:
        test_metrics_str += f"[{task.upper()}]{metrics_merged[task]} "
    print(f"\n{test_metrics_str}")

    print(f"\n✅ Inference completed!")
    print(f"   Average loss: {loss_avg:.4f}")

    return {
        "loss_avg": loss_avg,
        "metrics": {task: str(metrics_merged[task]) for task in model_tasks},
        "success_batches": success_batches,
        "failed_batches": failed_batches,
        "total_samples": total_samples
    }


def main():
    args = create_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    result = run_inference(
        args=args,
        device=device,
        server_url="http://124.17.4.221:30589/sft-temp-cls"
    )

    print("\n" + "=" * 60)
    print("✅ All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()