import copy
import json
import os
import sys

import numpy as np
import warnings
from absl import app, flags
import torch
from tensorboardX import SummaryWriter
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, CelebA, SVHN
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
from mia_evals.dataset_utils import MIACelebA, MIACIFAR10, MIACIFAR100, MIASVHN, MIASTL10, MIAImageFolder, MIAMNIST
from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score

from opacus import PrivacyEngine


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 30001, help='total training steps')  # 训练轮数
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', True, help='multi gpu training')

flags.DEFINE_string('dataset', 'CIFAR10', help='data set')
flags.DEFINE_string('dataset_root', './datasets', help='data set')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 0, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 2000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 10000, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 2000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_string('defense', 'none', help='Defense tricks, e.g., DP-SGD or L2 or CutOut')  # 防御方法
flags.DEFINE_bool('only_member', True, help='Training only on member split')
flags.DEFINE_string('ckpt_name', '', help='ckpt_name of check point(train) or target model(eval)')  # train：预加载模型衔接训练，eval：目标模型名称.pt
flags.DEFINE_float('dp_noise_multiplier', 1.1, help='noise multiplier used by DP-SGD')
flags.DEFINE_float('dp_max_grad_norm', 1.0, help='per-sample max grad norm used by DP-SGD')
flags.DEFINE_float('dp_delta', 1e-5, help='target delta used when reporting privacy budget')
flags.DEFINE_integer('dp_log_step', 1000, help='frequency of logging epsilon for DP-SGD, 0 to disable')
flags.DEFINE_float('l2_weight_decay', 5e-4, help='weight decay strength used when defense is L2')

# device = torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DP-SGD带来的封装问题
def remove_prefix(state_dict, prefix="_module."):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def is_dp_enabled():
    return FLAGS.defense.upper() == 'DP-SGD'


def evaluate(sampler, model, save_path, ckpt_name):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    # save sampled images
    torch.save({'samples': images}, os.path.join(save_path, 'sample', f'{ckpt_name}_samples.pt'))
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


class Cutout(object):
    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask


def get_dataset(FLAGS, only_member=False):
    dataset_root = FLAGS.dataset_root
    if FLAGS.dataset.upper() != 'CIFAR10-SYNTHETIC' and FLAGS.dataset.upper() != 'CIFAR10-GEN':
        print("real dataset")
        splits = np.load(f'./mia_evals/member_splits/{FLAGS.dataset.upper()}_train_ratio0.5.npz')
        member_idxs = splits['mia_train_idxs']
    else:
        member_idxs = None

    if FLAGS.dataset.upper() == 'CIFAR10':
        augmentations = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
        if 'cutout' in FLAGS.defense.lower():
            print('Applying CutOut augmentation')
            augmentations.append(Cutout(n_holes=1, length=8))
        augmentations.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5)))
        transforms = torchvision.transforms.Compose(augmentations)
        # transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
        #                                              torchvision.transforms.ToTensor(),
        #                                              torchvision.transforms.Normalize((0.5, 0.5, 0.5),
        #                                                                               (0.5, 0.5, 0.5))])
        if only_member:
            dataset = MIACIFAR10(member_idxs, root=os.path.join(dataset_root, 'cifar10'), train=True,
                                 transform=transforms, download=True)
        else:
            dataset = CIFAR10(root=os.path.join(dataset_root, 'cifar10'), train=True,
                              transform=transforms)
    elif FLAGS.dataset.upper() == 'CIFAR100':
        augmentations = [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
        if 'cutout' in FLAGS.defense.lower():
            print('Applying CutOut augmentation')
            augmentations.append(Cutout(n_holes=1, length=8))
        augmentations.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5)))
        transforms = torchvision.transforms.Compose(augmentations)
        # transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
        #                                              torchvision.transforms.ToTensor(),
        #                                              torchvision.transforms.Normalize((0.5, 0.5, 0.5),
        #                                                                               (0.5, 0.5, 0.5))])
        if only_member:
            dataset = MIACIFAR100(member_idxs, root=os.path.join(dataset_root, 'cifar100'), train=True,
                                  transform=transforms)
        else:
            dataset = CIFAR100(root=os.path.join(dataset_root, 'cifar100'), train=True,
                               transform=transforms)
    elif FLAGS.dataset.upper() == 'CELEBA':
        # for CelebA, first center crop 140 and then resize to 32 (by default)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(140),
            torchvision.transforms.Resize(FLAGS.img_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        if only_member:
            dataset = MIACelebA(member_idxs, root=os.path.join(dataset_root, 'celeba'), split='train',
                                transform=transforms, download=True)
        else:
            dataset = CelebA(root=os.path.join(dataset_root, 'celeba'), split='train',
                             transform=transforms, download=True)
    elif FLAGS.dataset.upper() == 'SVHN':
        transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                      (0.5, 0.5, 0.5))])
        if only_member:
            dataset = MIASVHN(member_idxs, root=os.path.join(dataset_root, 'svhn'), split='train',
                              transform=transforms, download=True)
        else:
            dataset = SVHN(root=os.path.join(dataset_root, 'svhn'), split='train',
                           transform=transforms, download=True)
    elif FLAGS.dataset.upper() == 'STL10_U':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        if only_member:
            dataset = MIASTL10(member_idxs, root=os.path.join(dataset_root, 'stl10'), split='unlabeled',
                               download=True, transform=transforms)
        else:
            dataset = torchvision.datasets.STL10(root=os.path.join(dataset_root, 'stl10'), split='unlabeled',
                                                 download=True, transform=transforms)
    elif FLAGS.dataset.upper() == 'TINY-IN':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        if only_member:
            dataset = MIAImageFolder(member_idxs, root=os.path.join(dataset_root, 'tiny-imagenet-200/train'),
                                     transform=transforms)
        else:
            dataset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_root, 'tiny-imagenet-200/train'),
                                                       transform=transforms)
    elif FLAGS.dataset.upper() == 'NWPU':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((FLAGS.img_size, FLAGS.img_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        nwpu_root = os.path.join(dataset_root, 'nwpu', 'NWPU-RESISC45')
        if only_member:
            dataset = MIAImageFolder(member_idxs, root=nwpu_root, transform=transforms)
        else:
            dataset = torchvision.datasets.ImageFolder(root=nwpu_root, transform=transforms)
    elif FLAGS.dataset.upper() == 'MNIST':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))
        ])
        if only_member:
            dataset  = MIAMNIST(member_idxs, root=os.path.join(dataset_root, 'mnist'), train=True,
                                 transform=transforms, download=True)
        else:
            dataset  = MIAMNIST(root=os.path.join(dataset_root, 'mnist'), train=True,
                                 transform=transforms, download=True)
            

    # 针对生成数据
    elif FLAGS.dataset.upper() == 'CIFAR10-GEN':
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])

        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(dataset_root, 'generated_cifar10/train'),
            transform=transforms
        )
    else:
        raise NotImplemented

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True,
                                              num_workers=FLAGS.num_workers)

    return data_loader


def train():
    dataloader = get_dataset(FLAGS, only_member=FLAGS.only_member)
    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    privacy_engine = None
    if FLAGS.defense == 'L2':
        print('Applying L2 Regularization')
        print(f'L2 weight decay = {FLAGS.l2_weight_decay}')
        optim = torch.optim.Adam(
            net_model.parameters(),
            lr=FLAGS.lr,
            weight_decay=FLAGS.l2_weight_decay,
        )
    elif is_dp_enabled():
        print('Applying DP-SGD using Opacus')
        optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
        privacy_engine = PrivacyEngine()
        net_model, optim, dataloader = privacy_engine.make_private(
            module=net_model,
            optimizer=optim,
            data_loader=dataloader,
            noise_multiplier=FLAGS.dp_noise_multiplier,
            max_grad_norm=FLAGS.dp_max_grad_norm,
        )
    else:
        optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)

    ema_model = copy.deepcopy(net_model)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        if is_dp_enabled():
            print('Disabling DataParallel because DP-SGD requires a single-device training path in this implementation')
        else:
            trainer = torch.nn.DataParallel(trainer)
            net_sampler = torch.nn.DataParallel(net_sampler)
            ema_sampler = torch.nn.DataParallel(ema_sampler)


    start_step = 0
    if FLAGS.ckpt_name != '':
        ckpt_path = os.path.join(FLAGS.logdir, FLAGS.ckpt_name)
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        net_model.load_state_dict(ckpt['net_model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        start_step = ckpt['step'] + 1
        print(f"Resuming training from step {start_step}")
    

    datalooper = infiniteloop(dataloader)

    # log setup
    if not os.path.exists(os.path.join(FLAGS.logdir, 'sample')):
        os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(start_step, FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            if not is_dp_enabled():
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # save
            if FLAGS.save_step > 0 and ((step+1) % FLAGS.save_step) == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, f'ckpt-step{step+1}.pt'))

            # sample
            if FLAGS.sample_step > 0 and ((step+1) % FLAGS.sample_step) == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % (step+1))
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()
                
            # evaluate
            if FLAGS.eval_step > 0 and ((step+1) % FLAGS.eval_step) == 0:
                ckpt_name = f"ckpt-step{step+1}"
                net_IS, net_FID, samples = evaluate(net_sampler, net_model, FLAGS.logdir, ckpt_name)
                save_image(
                    torch.tensor(samples[:256]),
                    os.path.join(FLAGS.logdir, 'sample', f'{ckpt_name}_samples.png'),
                    nrow=16)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model, FLAGS.logdir, f"ckpt-step{step+1}_ema")
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step+1, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step+1
                    f.write(json.dumps(metrics) + "\n")
            
            # DP-SGD
            if is_dp_enabled() and FLAGS.dp_log_step > 0 and ((step+1) % FLAGS.dp_log_step) == 0:
                epsilon = privacy_engine.get_epsilon(delta=FLAGS.dp_delta)
                privacy_log = {
                    'step': step + 1,
                    'epsilon': float(epsilon),
                    'delta': FLAGS.dp_delta,
                    'noise_multiplier': FLAGS.dp_noise_multiplier,
                    'max_grad_norm': FLAGS.dp_max_grad_norm,
                }
                print(
                    f"[Step {step+1}] epsilon = {epsilon:.2f}, "
                    f"delta = {FLAGS.dp_delta}, noise_multiplier = {FLAGS.dp_noise_multiplier}, "
                    f"max_grad_norm = {FLAGS.dp_max_grad_norm}"
                )
                with open(os.path.join(FLAGS.logdir, 'privacy.jsonl'), 'a') as f:
                    f.write(json.dumps(privacy_log) + "\n")
                
    writer.close()


def eval():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt_name = FLAGS.ckpt_name
    ckpt_path = os.path.join(FLAGS.logdir, ckpt_name)
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    
    state_dict = ckpt['net_model']
    print(FLAGS.defense)
    # 如果是 DP-SGD 训练出来的模型，则去除 '_module.' 前缀
    if is_dp_enabled():
        print("Detected DP-SGD: removing '_module.' prefix from state_dict keys")
        state_dict = remove_prefix(state_dict, prefix="_module.")

    model.load_state_dict(state_dict)
    (IS, IS_std), FID, samples = evaluate(sampler, model, FLAGS.logdir, ckpt_name.split('.')[0])
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'sample', f'{ckpt_name}_samples.png'),
        nrow=16)

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model, FLAGS.logdir, ckpt_name.split('.')[0])
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir,'sample', f'{ckpt_name}_samples_ema.png'),
        nrow=16)


# 生成图像用于循环训练扩散模型
def generate_synthetic_dataset(
    ckpt_path,
    save_root,
    num_images,
    batch_size=128,
):
    """
    Generate synthetic images using trained DDPM model.

    Output format (ImageFolder compatible):

    save_root/
        train/
            0/
            1/
            ...
            9/
    """

    print("Generating synthetic dataset...")
    print("Checkpoint:", ckpt_path)
    print("Num images:", num_images)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------- load model --------
    model = UNet(
        T=FLAGS.T,
        ch=FLAGS.ch,
        ch_mult=FLAGS.ch_mult,
        attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks,
        dropout=FLAGS.dropout
    ).to(device)

    sampler = GaussianDiffusionSampler(
        model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        FLAGS.img_size,
        FLAGS.mean_type,
        FLAGS.var_type
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt['net_model']

    # remove DP prefix if needed
    if is_dp_enabled():
        state_dict = remove_prefix(state_dict, "_module.")

    model.load_state_dict(state_dict)
    model.eval()

    # -------- prepare folders --------
    train_root = os.path.join(save_root, 'train')
    os.makedirs(train_root, exist_ok=True)

    img_id = 0

    with torch.no_grad():
        for _ in trange((num_images + batch_size - 1) // batch_size):

            cur_bs = min(batch_size, num_images - img_id)

            x_T = torch.randn(
                cur_bs, 3, FLAGS.img_size, FLAGS.img_size
            ).to(device)

            x_0 = sampler(x_T)
            x_0 = (x_0 + 1) / 2   # [-1,1] → [0,1]

            for i in range(cur_bs):
                label = np.random.randint(0, 10)

                class_dir = os.path.join(train_root, str(label))
                os.makedirs(class_dir, exist_ok=True)

                save_image(
                    x_0[i],
                    os.path.join(class_dir, f"{img_id:06d}.png")
                )

                img_id += 1

    print("Synthetic dataset saved to:", save_root)



def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
