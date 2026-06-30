from typing import Union
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from finetuneing.config import Config
from finetuneing.utils import *
from .postprocess import process_outputs,ResultSaver
import finetuneing.utils.help_builder as help_builder
from finetuneing.datasets.SFTData import get_loss
from finetuneing.utils.distributions import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# def outs_trans(x):
#     act_weight = nn.Sigmoid()
#     act_mu = nn.Sigmoid()
#     act_std = nn.ELU()
#     scale_factor = 8.0
#     assert x.shape[-1] % 3 == 0
#     num = x.shape[-1]//3
#     w = act_weight(x[:,:num]) - 0.5
#     mu = act_mu(x[:,num:2*num]) * scale_factor
#     std = act_std(x[:,2*num:]) + 1 + 1.0e-8
#     return torch.cat((w,mu,std),dim=-1)
        
def visualize_GT(data,save_path,gts,metrics_targets,data_name=None,outputs_for_loss=None,results=None,losses=None, ppks=None, step=0, mode='train'):
    if 'emg_z' not in metrics_targets.keys():
        return
    os.makedirs(save_path,exist_ok=True)
    torch.testing.assert_close(metrics_targets['emg_z'],gts.cpu())
    num_pic = 5

    preds = results['emg_z'].cpu().numpy()
    loss = torch.exp(-losses).cpu().detach().numpy()
    ppks = ppks.cpu().numpy()
    dist = GaussianMixture()

    if outputs_for_loss is not None:
        dist.set_params(outputs_for_loss)

    grid = torch.arange(-5, 10, 0.05).reshape(-1,1,1).to(dist.mus.device)
    with torch.no_grad():
        probs = dist.prob(grid)
    grid = grid.cpu().numpy()

    for id in range(num_pic): # visualize 10 each step
        emg_z = metrics_targets['emg_z'][id].cpu().numpy()
        x = data[id].cpu().numpy()
        prob = probs[:,id].cpu().numpy()
        plt.figure()
    
        fig, axes = plt.subplots(4, 1, figsize=(8, 8))

        # 在每个子图中绘制数据
        # 绘制竖线，表示p和s的值
        for i in range(3):
            axes[i].plot(x[i],label=f'mag: {emg_z[0]}')
            if ppks[id] > 0:
                axes[i].vlines(ppks[id],0,1,transform=axes[i].get_xaxis_transform(),colors='C1',label=f'{100 - ppks[id]/100.} sec')
            axes[i].set_title('channel {}'.format(i))
            axes[i].set_xlabel('Index')
            axes[i].set_ylabel('Value')
            axes[i].legend()
        axes[3].plot(grid.ravel(), prob)
        axes[3].vlines(emg_z[0],0,1,transform=axes[3].get_xaxis_transform(),colors='C1',label=f'True Mag: {emg_z[0]}')
        axes[3].vlines(preds[id,0],0,1,transform=axes[3].get_xaxis_transform(),colors='C2',label=f'Pred Mag: {preds[id,0]}')
        axes[3].hlines(loss[id],0,emg_z[0],linestyles="--",colors='C1',label=f'obs LL:{loss[id]}')
        axes[3].set_title('probability distribution')
        axes[3].set_xlabel('Mag')
        axes[3].set_ylabel('Likelihood')
        axes[3].legend()
            
        # 在画布上协商data_name
        if data_name:
            fig.suptitle(data_name[id])

        # 调整子图之间的间距
        plt.tight_layout()

        plt.close(fig)
        # 显示图形
        fig.savefig(os.path.join(save_path,f'_{id+step*num_pic:0>8}_{mode}.png'))

def validate(
    args, tasks,model, loss_fn, val_loader, epoch, device, testing=False
) -> Union[float, dict]:
    
    model.eval()
    
    model_labels, _ = help_builder.get_labels_tasks(args.downstream_task)
    tgts_trans_for_loss = None
    outs_trans_for_loss = None 
    outs_trans_for_res = None

    average_meters = {}
    metrics_merged = {}

    sampling_rate = val_loader.dataset.sampling_rate()
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
    if args.splitPS:
        average_meters["Pn_index_loss"] = AverageMeter("Pn_index_loss", ":6.4f")
        average_meters["Pg_index_loss"] = AverageMeter("Pg_index_loss", ":6.4f")
        average_meters["Sn_index_loss"] = AverageMeter("Sn_index_loss", ":6.4f")
        average_meters["Sg_index_loss"] = AverageMeter("Sg_index_loss", ":6.4f")
        average_meters["det_loss"] = AverageMeter("det_loss", ":6.4f")
    progress = ProgressMeter(
        len(val_loader),
        [m for m in average_meters.values()],
        prefix=f"{'Test' if testing else 'Val'}: [{epoch}/{args.epochs}]",
    )
    
    
    if testing and args.save_test_results and is_main_process():
        results_saver = ResultSaver(item_names=tasks)
    else:
        results_saver = None

    ana_dir = os.path.join(args.log_dir,'analysis')
    if is_main_process():
        if not os.path.exists(ana_dir):
            os.makedirs(ana_dir)
    #results_to_save = {'ppk':[],'ll':[],'true_mag':[],'pred_mag':[]}
    with torch.no_grad():
        for step, (x, loss_targets, metrics_targets, info_for_logging) in enumerate(val_loader):
            
            if isinstance(x, (list, tuple)):
                x = [xi.to(device) for xi in x]
            else:
                x = x.to(device)

            if isinstance(loss_targets, (list, tuple)):
                loss_targets = [yi.to(device) for yi in loss_targets]
            else:
                loss_targets = loss_targets.to(device)

            if 'ppks_type' in metrics_targets.keys() and 'spks_type' in metrics_targets.keys():
                outputs = model(x,metrics_targets['ppks_type'],metrics_targets['spks_type'])
            else:
                outputs = model(x)

            # Loss
            outputs_for_loss = outs_trans_for_loss(outputs) if outs_trans_for_loss is not None else outputs
            loss_targets = tgts_trans_for_loss(loss_targets) if tgts_trans_for_loss is not None else loss_targets
            if 'ppks_type' in metrics_targets.keys() and 'spks_type' in metrics_targets.keys():
                loss,loss_ = loss_fn(outputs_for_loss, loss_targets,show_loss=True)
                loss_dict = get_loss(loss_,metrics_targets['ppks_type'],metrics_targets['spks_type'])
                for key in loss_dict.keys():
                    average_meters[f"{key}_loss"].update(loss_dict[key][0],loss_dict[key][1])
            else:
                losses = loss_fn(outputs_for_loss, loss_targets)
                loss = losses.mean()

            # Batch size of this step
            if isinstance(x, (list, tuple)):
                step_batch_size = x[0].size(0)
            else:
                step_batch_size = x.size(0)

            # Reduce
            if is_dist_avail_and_initialized():
                loss = reduce_tensor(loss, "AVG")
                step_batch_size = torch.tensor(
                    step_batch_size, device=device, dtype=torch.int32
                )
                step_batch_size = reduce_tensor(step_batch_size)
                dist.barrier()
                step_batch_size = step_batch_size.item()

            # Save loss
            average_meters["loss"].update(loss.item(), step_batch_size)

            # Process outputs
            outputs_for_metrics = outs_trans_for_res(outputs) if outs_trans_for_res is not None else outputs
            results, lls = process_outputs(args, outputs_for_metrics,model_labels,sampling_rate)                

            #if step % 100 == 0 and is_main_process():
            #    visualize_GT(data=x,
            #                save_path=os.path.join(args.log_dir,f"plots/ep{epoch:0>2}"),
            #                gts=loss_targets,
            #                metrics_targets=metrics_targets,
            #                outputs_for_loss=outputs_for_loss,
            #                results=results,
            #                losses=losses,
            #                ppks=info_for_logging,
            #                step=step,
            #                mode='validate')
            #results_to_save['ppk'].append(info_for_logging.cpu().detach().numpy().ravel())
            #results_to_save['ll'].append(lls.cpu().detach().numpy().ravel())
            #results_to_save['true_mag'].append(metrics_targets['emg_z'].cpu().numpy().ravel())
            #results_to_save['pred_mag'].append(results['emg_z'].cpu().numpy().ravel())

            for task in tasks:
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
                    reduce=is_dist_avail_and_initialized(),
                )
                for metric in metrics.metric_names():
                    average_meters[f"{task}_{metric}"].update(
                        metrics.get_metric(name=metric), step_batch_size
                    )
                metrics_merged[f"{task}"].add(metrics)


            if is_main_process() and step % args.log_step == 0:
                prg_str = progress.get_str(batch_idx=step,name = f"{args.model_name}_{'test' if testing else 'val'}")
                print(prg_str)

#    for key in results_to_save.keys():
#        results_to_save[key] = np.concatenate(results_to_save[key])
#    df = pd.DataFrame(results_to_save)
#    fname = f'mag_res_{dist.get_rank():0>2}.csv'
#    df.to_csv(os.path.join(ana_dir,fname), index=False)
    if results_saver is not None:
        results_save_path = get_safe_path(os.path.join(args.log_dir,f"test_results_{val_loader.dataset.name()}.csv"))
        results_saver.save_as_csv(results_save_path)
    
    loss_avg = average_meters["loss"].avg
    return loss_avg, metrics_merged
