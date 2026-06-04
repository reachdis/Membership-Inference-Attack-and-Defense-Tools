import numpy as np
import random
import os

def preview_and_save_dataset(dataset_path, save_path, num_samples=5000, seed=0):
    random.seed(seed)

    data = np.load(dataset_path)
    train_data = data['mia_train_idxs']
    ratio = data['ratio']

    # 采样
    idxs = random.sample(sorted(train_data.tolist()), num_samples)
    idxs = np.array(idxs, dtype=np.int64)

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        selected_idxs=idxs,
        num_samples=num_samples,
        ratio=ratio,
        source_dataset=dataset_path
    )

    print(f"Saved {len(idxs)} indices to {save_path}")
    return idxs


dataset_path = './mia_evals/member_splits/CIFAR10_train_ratio0.5.npz'
save_path = './mia_evals/member_splits/CIFAR10_real_10k_idxs.npz'

preview_and_save_dataset(dataset_path, save_path, num_samples=10000)
