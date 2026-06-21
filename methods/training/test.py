import os
import torch
from finetuneing.config import Config
from finetuneing.models import load_checkpoint
from finetuneing.utils import *
from .preprocess import SeismicDataset,SFTDataset
from .postprocess import process_outputs
import finetuneing.utils.help_builder as help_builder

import LSD.models.backbone_ablation as backbone_ablation
from mup import set_base_shapes

from finetuneing.datasets.SFTData import get_loss
from .validate import visualize_GT

import numpy as np
import pandas as pd

#hanzr feature label
''' 
features = []
label_pt = []

# 定义钩子函数
def extract_features_hook(module, input, output):
    # out put module: output current layer; input: current layer input; output: current layer output
    print(f"Layer {module} output shape: {output.shape}")
    # save feature
    features.append(output)
'''

def test_worker(args, device)->float:
    if args.visualize and is_main_process():
        if args.visualize_save_dir == '':
            args.visualize_save_dir = os.path.join(args.log_dir, "test_visualize")
        if not os.path.exists(args.visualize_save_dir):
            os.makedirs(args.visualize_save_dir,exist_ok=True)
    # Data loader
    model_inputs = [['z', 'n', 'e']]
    model_labels, model_tasks = help_builder.get_labels_tasks(args.downstream_task)

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
        num_workers=args.workers,
    )

    # Load checkpoint
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, device=device)
        print(f"Model loaded: {args.checkpoint}")
    else:
        raise ValueError("checkpoint is None.")

    # Loss
    loss_fn = help_builder.get_loss(args.downstream_task, args.det_weight, args.p_weight, args.s_weight)
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
        # print("############### checkpoint keys: ", checkpoint.keys(), " model_dict keys: ", checkpoint["model_dict"].keys())
        model.load_state_dict(checkpoint["model_dict"], strict=False)
        print(f"model.load_state_dict")
    
    if is_main_process():
        print(f"Model parameters: {count_parameters(model)}")

    model = model.to(device)

    model.eval()
    model_labels, _ = help_builder.get_labels_tasks(args.downstream_task)
    tgts_trans_for_loss = None
    outs_trans_for_loss = None
    outs_trans_for_res = None

    average_meters = {}
    metrics_merged = {}

    sampling_rate = test_loader.dataset.sampling_rate()
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
    if args.splitPS:
        average_meters["Pn_index_loss"] = AverageMeter("Pn_index_loss", ":6.4f")
        average_meters["Pg_index_loss"] = AverageMeter("Pg_index_loss", ":6.4f")
        average_meters["Sn_index_loss"] = AverageMeter("Sn_index_loss", ":6.4f")
        average_meters["Sg_index_loss"] = AverageMeter("Sg_index_loss", ":6.4f")
        average_meters["det_loss"] = AverageMeter("det_loss", ":6.4f")
        
    progress = ProgressMeter(
        len(test_loader),
        [m for m in average_meters.values()],
        prefix=f"{'Test'}:",
    )

    '''
    conf_metrix = []
    label_pt = []
    # pred -- true
    lplp = 0
    lpvt = 0
    lpns = 0
    vtlp = 0
    vtvt = 0
    vtns = 0
    nslp = 0
    nsvt = 0
    nsns = 0
    '''

    ana_dir = os.path.join(args.log_dir,'analysis')
    #results_to_save = {'ppk':[],'ll':[],'true_mag':[],'pred_mag':[]}

    #hanzr 
    num_layers = sum(1 for _ in model.modules())
    print(f"Total number of layers in the model: {num_layers}")
    print(model[-2])

    # hook_handle = model[-2].register_forward_hook(extract_features_hook)  # model[-3] 是倒数第 3 层



    with torch.no_grad():
        wrong_wave_list = []
        print(len(test_loader))
        for step, (x, loss_targets, metrics_targets, info_for_logging) in enumerate(test_loader):
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

            #hanzr 
            # print(f"Step {step}, Extracted features shape:", features[0].shape)


            
            # save wrong predicted wavedata and get confusion metrix
            for i in range(loss_targets.shape[0]):
                pred_cls = torch.argmax(outputs[i])
                true_cls = torch.argmax(loss_targets[i])
                # hanzr
                # label_pt.append(true_cls)
                '''
                if pred_cls == 0:
                    if true_cls == 0:
                        lplp += 1
                    elif true_cls == 1:
                        lpvt += 1
                    else:
                        lpns += 1
                elif pred_cls == 1:
                    if true_cls == 0:
                        vtlp += 1
                    elif true_cls == 1:
                        vtvt += 1
                    else:
                        vtns += 1
                elif pred_cls == 2:
                    if true_cls == 0:
                        nslp += 1
                    elif true_cls == 1:
                        nsvt += 1
                    else:
                        nsns += 1
                if pred_cls == true_cls:
                    continue
                else:
                    cur = torch.tensor([pred_cls, true_cls])
                    label_pt.append(cur)
                    wrong_wave_list.append(x[i])
            '''

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
            step_batch_size = x.size(0)

            # Save loss
            average_meters["loss"].update(loss.item(), step_batch_size)

            # Process outputs
            outputs_for_metrics = outs_trans_for_res(outputs) if outs_trans_for_res is not None else outputs
            results, lls = process_outputs(args, outputs_for_metrics,model_labels,sampling_rate)                

            #if step % 100 == 0 and is_main_process():
            #    visualize_GT(data=x,
            #                save_path=os.path.join(args.log_dir,"plots/test"),
            #                gts=loss_targets,
            #                metrics_targets=metrics_targets,
            #                outputs_for_loss=outputs_for_loss,
            #                results=results,
            #                losses=losses,
            #                ppks=info_for_logging,
            #                step=step,
            #                mode='test')
            #results_to_save['ppk'].append(info_for_logging.cpu().detach().numpy().ravel())
            #results_to_save['ll'].append(lls.cpu().detach().numpy().ravel())
            #results_to_save['true_mag'].append(metrics_targets['emg_z'].cpu().numpy().ravel())
            #results_to_save['pred_mag'].append(results['emg_z'].cpu().numpy().ravel())
            
            # hanzr
            # torch.save(features, '/public/home/zhangbei/work_dir/kangkai/proj_han/output/features_SNE/features.pt')
            # torch.save(label_pt, '/public/home/zhangbei/work_dir/kangkai/proj_han/output/features_SNE/labels.pt')

            if is_main_process() and args.visualize and step % 50 == 0 and 'ppk' in model_tasks:
                # Only applicable to phase-picking task.
                vis_waves_preds_targets(x[0].detach().cpu().numpy(),
                                        outputs[0].detach().cpu().numpy(),
                                        loss_targets[0].detach().cpu().numpy(),
                                        sampling_rate,
                                        args.visualize_save_dir,
                                        step_epoch=(step,-1))
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


            if is_main_process() and step % args.log_step == 0:
                prg_str = progress.get_str(batch_idx=step,name = f"{args.model_name}_{'test'}")
                print(prg_str)
        
        '''
        conf_metrix = [lplp, lpvt, lpns, vtlp, vtvt, vtns, nslp, nsvt, nsns]
        print('confusion metrix: ')
        print(conf_metrix)
        torch.save(wrong_wave_list, '/public/home/zhangbei/work_dir/kangkai/proj_han/output/6976TrainSet/5e-1/wrong_predicted_data_waveform.pt')
        torch.save(label_pt, '/public/home/zhangbei/work_dir/kangkai/proj_han/output/6976TrainSet/5e-1/wrong_predicted_data_label_pred&true.pt')
        '''
    

#    for key in results_to_save.keys():
#        results_to_save[key] = np.concatenate(results_to_save[key])
#    df = pd.DataFrame(results_to_save)
#    fname = f'mag_res_test.csv'
#    df.to_csv(os.path.join(ana_dir,fname), index=False)
    loss_avg = average_meters["loss"].avg

    if is_main_process():
        # Metrics merged
        test_metrics_str = "* "
        for task in model_tasks:
            test_metrics_str += f"[{task.upper()}]{metrics_merged[task]} "
        print(test_metrics_str)


        #hanzr
    # hook_handle.remove()
    
    return loss_avg
