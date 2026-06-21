import requests
import json

config = {
    # test or fft
    "checkpoint": "/public/home/zhangbei/work_dir/kangkai/proj_han/output/new/6976/checkpoints/model-latest.pth",
    "hps": "scaling_mae_mup_ablation/transfered/HPs_schedulecosine_ep100_data400000_modelsize512_lr0.001_wd0.0005_mr0.9_init-std0.08_attn-mult32.0_input-mult14.142135623730951_output-mult2.0",
    "seed": 0,
    "mode": "test",
    "model-name": "llama",
    "downstream-task": "cls",
    "target-distribution-name": "GaussianMixture",
    "log-dir": "/public/home/zhangbei/work_dir/kangkai/proj_han/output/tsne_model",
    "data-split": True,
    "train-size": 0.9,
    "val-size": 0.1,
    "shuffle": False,
    "workers": 1,
    "in-samples": 10000,
    "patch_size": 50,
    "augmentation": True,
    "pre-emphasis-rate": 0.0,
    "add-gap-rate": 0.0,
    "band-filt": False,
    "p-position-ratio": 0.02,
    "p_position_ratio_range_or_sigma": 0.02,
    "p_position_ratio_type": "uniform",
    "epochs": 90,
    "patience": 300,
    "batch-size": 50,
    "warmup-steps": 0.02,
    "optim": "adamw",
    "weight_decay": 0.05,
    "base-lr": 0.01,
    "lr-scheduler": "cosine",
    "pretrain-method": "lp",
    "encoder_size": "proxy",
    "eval_type": "finetune",
    "pool_type": "avg",
    "log-step": 10,
    "subset_names": "cls_data",
    "train_meta_data_path": "/public/home/zhangbei/work_dir/kangkai/prepared_data/splited_data/lp_vt_train_7k.csv",
    "test_meta_data_path": "/public/home/zhangbei/work_dir/kangkai/prepared_data/splited_data/lp_vt_test_7k.csv",
    "train_data_dir": "/public/home/zhangbei/work_dir/kangkai/prepared_data/lp_vt.hdf5",
    "test_data_dir": "/public/home/zhangbei/work_dir/kangkai/prepared_data/lp_vt.hdf5",
    "train_sample_num": 6975
}

print("=" * 60)
print("Sending request to /cls endpoint...")
print("=" * 60)

try:
    response = requests.post(
        "http://124.17.4.220:10089/cls",
        json=config,
        timeout=3600
    )

    if response.status_code == 200:
        print("✅ Success!")
        data = response.json()
        
        print(f"Mode: {data.get('mode')}")
        print(f"Device: {data.get('device')}")
        print(f"Accuracy: {data.get('accuracy_formatted')}")
        print("=" * 60)
        
        output = data.get('output', {})
        combined_output = output.get('combined', '')
        
        if combined_output:
            print("📋 Full output:")
            print(combined_output)
        else:
            print("No output captured")
            
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except requests.exceptions.Timeout:
    print("⏰ Request timed out (training may still be running)")
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to server. Is it running?")
except Exception as e:
    print(f"❌ Unexpected error: {e}")