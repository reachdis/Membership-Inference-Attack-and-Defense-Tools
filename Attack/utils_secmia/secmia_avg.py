import argparse
import json
import os
import numpy as np
import torch

import secmia as base_secmi
from mia_evals.dataset_utils import load_member_data


def summarize_metric(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "all": [float(x) for x in arr.tolist()],
    }


def build_model(ckpt_path, flags_obj, model_type):
    if model_type == "ddpm":
        return base_secmi.get_model(ckpt_path, flags_obj, WA=True).to(base_secmi.device)
    if model_type == "etdm":
        return base_secmi.get_model_grouped(ckpt_path, flags_obj).to(base_secmi.device)
    if model_type == "bwcd":
        return base_secmi.get_model_batched(ckpt_path, flags_obj).to(base_secmi.device)
    if model_type == "hcmd":
        return base_secmi.get_model_heuristic(ckpt_path, flags_obj).to(base_secmi.device)
    if model_type == "gcmd":
        return base_secmi.get_model_heuristic_grouped(ckpt_path, flags_obj).to(base_secmi.device)
    if model_type == "smcd":
        return base_secmi.get_model_selective(ckpt_path, flags_obj).to(base_secmi.device)
    raise ValueError(f"Unknown model_type: {model_type}")


def collect_t_results(model, flags_obj, data_root, dataset, t_sec, timestep, batch_size):
    _, _, member_loader, nonmember_loader = load_member_data(
        dataset_root=data_root,
        dataset_name=dataset,
        batch_size=batch_size,
        shuffle=False,
        randaugment=False,
    )

    member_results = base_secmi.get_intermediate_results(model, flags_obj, member_loader, t_sec, timestep)
    nonmember_results = base_secmi.get_intermediate_results(model, flags_obj, nonmember_loader, t_sec, timestep)
    return {
        "member_diffusions": member_results["internal_diffusions"],
        "member_internal_samples": member_results["internal_denoise"],
        "nonmember_diffusions": nonmember_results["internal_diffusions"],
        "nonmember_internal_samples": nonmember_results["internal_denoise"],
    }


def save_avg_results(result_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(result_dict, save_path)
    print(f"[✓] Average SecMI results saved to: {save_path}")


def append_log_line(log_path, line):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def run_avg_for_ckpt(args, ckpt_name):
    print("=" * 60)
    print(f"[✓] Averaging SecMI for model: {ckpt_name}")
    print("=" * 60)

    ckpt_path = os.path.join(args.model_dir, ckpt_name)
    flag_path = os.path.join(args.model_dir, "flagfile.txt")

    flags_obj = base_secmi.get_FLAGS(flag_path)
    flags_obj(["secmia_avg.py"])
    flags_obj.ckpt_name = ckpt_name
    flags_obj.output_save_dir = args.output_save_dir

    log_root = os.path.join(args.output_save_dir, "SecMI_attack_results_avg")
    stem = os.path.splitext(ckpt_name)[0]
    log_path = os.path.join(log_root, f"SecMI_AVG_{stem}_{args.t_sec}.log")

    append_log_line(log_path, "=" * 72)
    append_log_line(log_path, f"Model: {ckpt_name}")
    append_log_line(log_path, f"Model type: {args.model_type}")
    append_log_line(log_path, f"Dataset: {args.dataset}")
    append_log_line(log_path, f"t_sec={args.t_sec}, m={args.m}, n_runs={args.n_runs}, seed={args.seed}")
    append_log_line(log_path, "-" * 72)

    model = build_model(ckpt_path, flags_obj, args.model_type)
    t_results = collect_t_results(
        model=model,
        flags_obj=flags_obj,
        data_root=args.data_root,
        dataset=args.dataset,
        t_sec=args.t_sec,
        timestep=args.m,
        batch_size=args.batch_size,
    )

    print("[Info] Running deterministic statistic attack once...")
    stat_result = base_secmi.execute_attack(t_results, type="stat")
    append_log_line(
        log_path,
        (
            f"STAT single: ASR={float(stat_result['asr']):.6f}, "
            f"AUC={float(stat_result['auc']):.6f}"
        ),
    )

    print(f"[Info] Running NNS attack {args.n_runs} times...")
    nns_runs = []
    for run_idx in range(args.n_runs):
        seed = args.seed + run_idx
        print(f"\n-------- NNS Run {run_idx + 1}/{args.n_runs} | seed={seed} --------")
        base_secmi.fix_seed(seed)
        run_result = base_secmi.execute_attack(t_results, type="nns")
        nns_runs.append(run_result)
        append_log_line(
            log_path,
            (
                f"NNS run {run_idx + 1:02d}: seed={seed}, "
                f"ASR={float(run_result['asr']):.6f}, "
                f"AUC={float(run_result['auc']):.6f}"
            ),
        )

    stat_summary = {
        "auc": {"mean": float(stat_result["auc"]), "std": 0.0, "all": [float(stat_result["auc"])]},
        "asr": {"mean": float(stat_result["asr"]), "std": 0.0, "all": [float(stat_result["asr"])]},
        "TPR@1%FPR": {
            "mean": float(stat_result["TPR@1%FPR"]),
            "std": 0.0,
            "all": [float(stat_result["TPR@1%FPR"])],
        },
        "TPR@0.1%FPR": {
            "mean": float(stat_result["TPR@0.1%FPR"]),
            "std": 0.0,
            "all": [float(stat_result["TPR@0.1%FPR"])],
        },
    }

    nns_summary = {
        "auc": summarize_metric([run["auc"] for run in nns_runs]),
        "asr": summarize_metric([run["asr"] for run in nns_runs]),
        "TPR@1%FPR": summarize_metric([float(run["TPR@1%FPR"]) for run in nns_runs]),
        "TPR@0.1%FPR": summarize_metric([float(run["TPR@0.1%FPR"]) for run in nns_runs]),
    }

    print("\n[Stat] Mean Results")
    print(json.dumps({k: v["mean"] for k, v in stat_summary.items()}, indent=2))

    print("\n[NNS] Mean Results")
    print(json.dumps({k: v["mean"] for k, v in nns_summary.items()}, indent=2))
    print("[NNS] Std Results")
    print(json.dumps({k: v["std"] for k, v in nns_summary.items()}, indent=2))

    append_log_line(log_path, "-" * 72)
    append_log_line(
        log_path,
        (
            f"STAT mean: ASR={stat_summary['asr']['mean']:.6f}, "
            f"AUC={stat_summary['auc']['mean']:.6f}"
        ),
    )
    append_log_line(
        log_path,
        (
            f"NNS mean: ASR={nns_summary['asr']['mean']:.6f} "
            f"(std={nns_summary['asr']['std']:.6f}), "
            f"AUC={nns_summary['auc']['mean']:.6f} "
            f"(std={nns_summary['auc']['std']:.6f})"
        ),
    )
    append_log_line(log_path, "=" * 72)

    save_root = log_root
    save_path = os.path.join(save_root, f"SecMI_AVG_{stem}_{args.t_sec}.pt")
    payload = {
        "ckpt_name": ckpt_name,
        "model_type": args.model_type,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "log_path": log_path,
        "stat_single": stat_result,
        "stat_summary": stat_summary,
        "nns_runs": nns_runs,
        "nns_summary": nns_summary,
    }
    save_avg_results(payload, save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./logs/ETDM_CIFAR10")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--t_sec", type=int, default=100)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num_t_groups", type=int, default=5)
    parser.add_argument("--model_type", type=str, default="ddpm")
    parser.add_argument("--ckpt_names", nargs="+", default=["ckpt-step8000.pt"])
    parser.add_argument("--output_save_dir", type=str, default="./experiment_results/recent")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_secmi.define_flags()
    for ckpt_name in args.ckpt_names:
        run_avg_for_ckpt(args, ckpt_name)
