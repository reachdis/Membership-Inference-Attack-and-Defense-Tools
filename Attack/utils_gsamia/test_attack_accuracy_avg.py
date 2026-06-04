import torch
import numpy as np
import os
import argparse
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Test attack accuracy.")
    parser.add_argument(
        "--gsa", 
        default=1,
    )
    parser.add_argument(
        "--target_model_member_path", 
        default='./gradient_mia/target_feature_ETCD_k_3/mem/gradient_ckpt-step400000.pt',
    )
    parser.add_argument(
        "--target_model_non_member_path", 
        default='./gradient_mia/target_feature_ETCD_k_3/nonmem/gradient_ckpt-step400000.pt',
    )
    parser.add_argument(
        "--shadow_model_member_path", 
        nargs='+', 
        help='<Required> Set flag: e.g. --shadow_model_member_path ./shadow_feature_DDPM/mem/gradient_ckpt-step150000.pt', 
        required=True
    )
    parser.add_argument(
        "--shadow_model_non_member_path", 
        nargs='+', 
        help='<Required> Set flag', 
        required=True
    )

    return parser.parse_args()


def load_shadow_data():
    shadow_model_member_list = []
    shadow_model_non_member_list = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for shadow_mem, shadow_non_mem in zip(args.shadow_model_member_path, args.shadow_model_non_member_path):
        shadow_model_member_list.append(torch.load(shadow_mem, map_location=device).to(device))
        shadow_model_non_member_list.append(torch.load(shadow_non_mem, map_location=device).to(device))
    shadow_member = torch.cat(shadow_model_member_list, dim = 0)
    shadow_non_member = torch.cat(shadow_model_non_member_list, dim = 0)
    return shadow_member, shadow_non_member

def load_target_data(member_path, non_member_path):
    target_member = torch.load(member_path, map_location=device)
    target_non_member = torch.load(non_member_path, map_location=device)
    return target_member, target_non_member

def preprocess(member, non_member):
    train_np = member.cpu().numpy()
    test_np = non_member.cpu().numpy()
    train_np = train_np[0:test_np.shape[0]]
    train_y_np = np.ones(train_np.shape[0])
    test_y_np = np.zeros(test_np.shape[0])
    x = np.vstack((train_np, test_np))
    y = np.concatenate((train_y_np, test_y_np))
    x = preprocessing.scale(x)
    return x, y

if __name__ == "__main__":
    args = parse_args()

    # ========= 实验重复次数 =========
    N_RUNS = 10  # 可根据需要修改
    print(f"\n===== Running GSA attack for {N_RUNS} runs =====\n")

    # 加载 shadow 与 target 数据只需一次
    shadow_member, shadow_non_member = load_shadow_data()
    shadow_x, shadow_y = preprocess(shadow_member, shadow_non_member)

    target_member, target_non_member = load_target_data(
        args.target_model_member_path, 
        args.target_model_non_member_path
    )
    target_x, target_y = preprocess(target_member, target_non_member)

    # ====== 统计多次结果 ======
    all_asr = []
    all_auc = []
    all_tpr_01 = []
    all_tpr_001 = []

    for run in range(N_RUNS):
        print(f"\n======== Run {run+1} / {N_RUNS} ========")

        # ---- shadow train-test split ----
        shadow_train_x, shadow_test_x, shadow_train_y, shadow_test_y = \
            train_test_split(shadow_x, shadow_y, test_size=0.3, shuffle=True)

        # ---- 训练 XGB ----
        print("Training XGB...")
        xgb = XGBClassifier(n_estimators=200)
        xgb.fit(shadow_train_x, shadow_train_y)

        # ---- attack on target ----
        pred_label = xgb.predict(target_x)
        pred_prob = xgb.predict_proba(target_x)

        # ASR
        asr = np.mean(pred_label == target_y)

        # ROC-AUC
        auc = roc_auc_score(target_y, pred_prob[:, 1])

        # TPR-FPR
        fpr, tpr, thresholds = roc_curve(target_y, pred_prob[:, 1])
        tpr_01 = tpr[np.argmin(np.abs(fpr - 0.01))]
        tpr_001 = tpr[np.argmin(np.abs(fpr - 0.001))]

        # ---- 保存本次结果 ----
        all_asr.append(asr)
        all_auc.append(auc)
        all_tpr_01.append(tpr_01)
        all_tpr_001.append(tpr_001)

        print(f"Run {run+1} ASR = {asr:.4f}, AUC = {auc:.4f}, TPR@0.01 = {tpr_01:.4f}, TPR@0.001 = {tpr_001:.4f}")

    # ======== 输出平均结果 ========
    mean_asr = np.mean(all_asr)
    mean_auc = np.mean(all_auc)
    mean_tpr_01 = np.mean(all_tpr_01)
    mean_tpr_001 = np.mean(all_tpr_001)

    print("\n======== Final Mean Results ========")
    print(f"Mean ASR   = {mean_asr:.4f}  (std={np.std(all_asr):.4f})")
    print(f"Mean AUC   = {mean_auc:.4f}  (std={np.std(all_auc):.4f})")
    print(f"Mean TPR@FPR=0.01  = {mean_tpr_01:.4f}")
    print(f"Mean TPR@FPR=0.001 = {mean_tpr_001:.4f}")

    # ====== 构建保存路径（保持你原来的逻辑） ======
    target_member_path = args.target_model_member_path
    parts = target_member_path.split('/')
    dataset_name = None
    for part in parts:
        if part.startswith("gradient_mia_new"):
            dataset_name = part.replace("gradient_mia_new", "")
            break

    model_dir = os.path.basename(os.path.dirname(os.path.dirname(target_member_path)))
    model_type = model_dir.replace('target_feature_', '')
    feature_filename = os.path.basename(target_member_path)
    feature_name = os.path.splitext(feature_filename)[0]

    model_type_dataset = f"{model_type}_{dataset_name}" if dataset_name else model_type
    print(model_type_dataset)
    save_dir = os.path.join("experiment_results", model_type_dataset, "GSA_attack_results")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"GSA{args.gsa}_{feature_name}_mean.pt")

    # 保存平均结果
    results = {
        "mean_asr": mean_asr,
        "mean_auc": mean_auc,
        "mean_tpr@0.01": mean_tpr_01,
        "mean_tpr@0.001": mean_tpr_001,
        "all_asr": all_asr,
        "all_auc": all_auc,
        "all_tpr@0.01": all_tpr_01,
        "all_tpr@0.001": all_tpr_001,
    }

    torch.save(results, save_path)
    print(f"\n[✓] Mean GSA attack results saved to: {save_path}")
