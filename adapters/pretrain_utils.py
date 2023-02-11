import numpy as np 
import torch 
import shutil
import torch.nn.functional as F
import torchvision
import math
import time
import datasets
# from r2score import r2_score
import math
from typing import Tuple
from torch import Tensor
from torchvision.transforms import functional as FT
import wandb
from tsa_resnet import sample_adapter_configuration, TSA_Conv2d

def train(train_loader, model, criterion, optimizer, scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    ce_losses = AverageMeter('CE Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, ce_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    iters_per_epoch = len(train_loader)
    acc1_avg, acc5_avg = 0.0, 0.0
    
    for iter, data in enumerate(train_loader):
        images, labels = data[0], data[1]
        data_time.update(time.time() - end)
        learning_rates.update(optimizer.param_groups[0]['lr'])
        if args.gpu is not None:
            # Size of images is args.batch_size * (args.num_augs + 1)
            images = images.reshape(-1, 3, datasets.img_size, datasets.img_size).cuda(args.gpu, non_blocking=True) 
            labels = labels.cuda(args.gpu, non_blocking=True)

        # Sample model subnet = adapter configuration
        model.sample_subnet()
        with torch.cuda.amp.autocast(False):
            output, _= model(images)
            loss = criterion(output, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        acc1_avg += acc1.item()
        acc1_avg += acc5.item()
        wandb.log({"acc": acc1, "ce loss": loss})
        ce_losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))
    
        batch_time.update(time.time() - end)
        end = time.time()
        # torch.cuda.empty_cache()
        if iter % args.print_freq == 0:
            progress.display(iter)

    acc1_avg /= (iter+1)
    acc5_avg /= (iter+1)
    torch.cuda.empty_cache()
   
    return acc1_avg, acc5_avg, ce_losses.avg


def evaluate(train_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    ce_losses = AverageMeter('CE Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, ce_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    iters_per_epoch = len(train_loader)
    acc1_avg, acc5_avg = 0.0, 0.0
    similarity_matrix = 0.0
    all_accuracies = np.zeros(args.n_eval_adapters)

    # Sample N random configurations
    eval_adapter_configurations = []
    num_tsa_layers = len([n for n, m in model.named_modules() if isinstance(m, TSA_Conv2d)])
    for k in range(args.n_eval_adapters):
        adapter_configuration = np.array([sample_adapter_configuration(args.num_adapters) for _ in range(num_tsa_layers)])
        eval_adapter_configurations.append(adapter_configuration)

    for iter, data in enumerate(train_loader):
        acc1_list = np.zeros(args.n_eval_adapters)
        images, labels = data[0], data[1]
        data_time.update(time.time() - end)
        if args.gpu is not None:
            # Size of images is args.batch_size * (args.num_augs + 1)
            images = images.reshape(-1, 3, datasets.img_size, datasets.img_size).cuda(args.gpu, non_blocking=True) 
            labels = labels.cuda(args.gpu, non_blocking=True)

        images = get_aug_wise_images(images, args)
    
        loss = 0.0 # Mean loss of adapter configuration
        diff = 0.0
        features = []

        for k in range(args.n_eval_adapters):
            model.sample_subnet(eval_adapter_configurations[k])
            with torch.no_grad():
                logits, feats = model(images)
                orig_image_logits = logits[args.batch_size * args.num_augs:, :]
                loss += criterion(orig_image_logits, labels)
                acc1, acc5 = accuracy(orig_image_logits, labels, topk=(1, 5))
                features.append(feats)
                acc1_list[k] = acc1.item()
        features = torch.cat(features).view(len(features), images.shape[0], -1)
        similarity_matrix += get_similarity_vector(features, args)
        diff += get_pairwise_rowdiff(similarity_matrix)
        

        wandb.log({"acc": np.mean(acc1_list), "ce loss": loss})
        ce_losses.update(loss.item()/(k+1), images.size(0))
        top1.update(np.mean(acc1_list), images.size(0))

        all_accuracies += acc1_list
        batch_time.update(time.time() - end)
        end = time.time()
        # torch.cuda.empty_cache()
        if iter % args.print_freq == 0:
            progress.display(iter)


    all_accuracies /= (iter+1)
    similarity_matrix /= (iter+1)
    diff /= (iter+1)
    return all_accuracies, similarity_matrix, diff, eval_adapter_configurations

def get_pairwise_rowdiff(sim_matrix, criterion = torch.nn.L1Loss()):
    diff = 0.0
    for i in range(0, sim_matrix.shape[0]):
        for j in range(i+1, sim_matrix.shape[0]):
            diff += torch.exp(-criterion(sim_matrix[i], sim_matrix[j]))
    return diff

def get_similarity_vector(feats, args):
    # returns N vectors of R^k, each element being the similarity between original and augmented image
    sim = torch.zeros((args.n_eval_adapters, args.num_augs)).cuda(args.gpu)

    # Unaugmented images 
    orig_feats = feats[:, args.batch_size * args.num_augs:, :]

    for i in range(0, args.num_augs):
        aug_feats = feats[:, args.batch_size * i: args.batch_size * (i+1), :]
        sim[:, i] = F.cosine_similarity(orig_feats, aug_feats, dim = 2).mean(1)
    return sim

def get_aug_wise_images(images, args):
    images_list = []
    for i in range(args.num_augs+1):
        l = images[i::args.num_augs+1, :, :, :]
        images_list.append(l)
    images_list = torch.cat(images_list, dim = 0)
    return images_list


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def adjust_learning_rate(optimizer, epoch, args, prev_lr, linear_gamma = 0.16, decay_gamma = 0.975):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        # lr = prev_lr + linear_gamma
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = prev_lr * decay_gamma
        # lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Please provide a valid positive value for the num_classes. Got num_classes={num_classes}"
            )

        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("Please provide a valid positive value for the num_classes.")
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        _, H, W = FT.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar') 