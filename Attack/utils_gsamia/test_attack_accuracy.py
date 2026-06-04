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
    shadow_member, shadow_non_member = load_shadow_data()
    shadow_x, shadow_y = preprocess(shadow_member, shadow_non_member)
    shadow_train_x, shadow_test_x, shadow_train_y, shadow_test_y = train_test_split(shadow_x, shadow_y, test_size = 0.3)
    print("Training XGB...")
    xgb = XGBClassifier(n_estimators=200)
    xgb.fit(shadow_train_x, shadow_train_y)
    pred_xgb = xgb.predict(shadow_train_x)
    print("Shadow Train Results -------------------------------")
    print("XGBoost Classification Report=\n\n", classification_report(shadow_train_y, pred_xgb)) 
    print("XGBoost Confusion Matrix=\n\n", confusion_matrix(shadow_train_y, pred_xgb)) 

    pred_xgb = xgb.predict(shadow_test_x)
    print("Shadow Test Results -------------------------------")
    print("XGBoost Classification Report=\n\n", classification_report(shadow_test_y, pred_xgb)) 
    print("XGBoost Confusion Matrix=\n\n", confusion_matrix(shadow_test_y, pred_xgb)) 
    
    target_member, target_non_member = load_target_data(args.target_model_member_path, args.target_model_non_member_path)
    target_x, target_y = preprocess(target_member, target_non_member)
    pred_xgb = xgb.predict(target_x)
    print("Target Attack Results -------------------------------")
    print("XGBoost Classification Report=\n\n", classification_report(target_y, pred_xgb,digits = 3)) 
    print("XGBoost Confusion Matrix=\n\n", confusion_matrix(target_y, pred_xgb))
    # 计算 ASR（成员和非成员都预测正确的比例）
    asr = np.mean(pred_xgb == target_y)
    print(f"ASR (Overall Attack Success Rate): {asr:.4f}")

    pred_xgb = xgb.predict_proba(target_x)
    roc_auc = roc_auc_score(target_y, pred_xgb[:,1])

    print(f"ROC AUC: {roc_auc}")
    
    fpr, tpr, thresholds = roc_curve(target_y, pred_xgb[:,1])

    desired_fpr = 0.01

    closest_fpr_index = np.argmin(np.abs(fpr - desired_fpr))
    tpr_at_desired_fpr = tpr[closest_fpr_index]

    print(f"TPR at FPR = {desired_fpr}: {tpr_at_desired_fpr}")

    desired_fpr = 0.001

    closest_fpr_index = np.argmin(np.abs(fpr - desired_fpr))
    tpr_at_desired_fpr = tpr[closest_fpr_index]

    print(f"TPR at FPR = {desired_fpr}: {tpr_at_desired_fpr}")

    # 攻击结果保存
    # 解析目标模型路径
    target_member_path = args.target_model_member_path
    # parts = target_member_path.split(os.sep)
    parts = target_member_path.split('/')
    # 自动识别数据集或防御方法名称，如 gradient_mia_CIFAR100 → CIFAR100
    dataset_name = None
    for part in parts:
        if part.startswith("gradient_mia_new"):
            dataset_name = part.replace("gradient_mia_new", "")
            break
    model_dir = os.path.basename(os.path.dirname(os.path.dirname(target_member_path)))  # 'target_feature_ETCD_k_3'
    model_type = model_dir.replace('target_feature_', '')                               # 'ETCD_k_3'
    feature_filename = os.path.basename(target_member_path)             # 'gradient_ckpt-step250000.pt'
    feature_name = os.path.splitext(feature_filename)[0]                # 'gradient_ckpt-step250000'

    # 拼接 model_type_dataset 形式
    model_type_dataset = f"{model_type}_{dataset_name}" if dataset_name else model_type

    # 构建保存路径
    save_dir = os.path.join("experiment_results", model_type_dataset, "GSA_attack_results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"GSA{args.gsa}_{feature_name}.pt")

    # 保存结果字典
    results = {
        "roc_auc": roc_auc,
        "asr": asr,
        "tpr@fpr=0.01": tpr[np.argmin(np.abs(fpr - 0.01))],
        "tpr@fpr=0.001": tpr[np.argmin(np.abs(fpr - 0.001))],
        "target_y": target_y,
        "pred_proba": pred_xgb[:, 1],
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }

    torch.save(results, save_path)
    print(f"\n[✓] GSA attack results saved to: {save_path}")