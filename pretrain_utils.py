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

def train(train_loader, sup_train_loader, models_ensemble, criterion, optimizer, scaler, model_ema, mixupcutmix, epoch, args, running_mean, coeff, kl_coeff):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    ce_losses = AverageMeter('CE Loss', ':.4e')
    variance = AverageMeter('Variance', ':.4e')
    kl = AverageMeter('KL Div', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, ce_losses, variance, kl, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    models_ensemble.train()
    end = time.time()
    avg_sim = 0.0
    iters_per_epoch = len(train_loader)
    acc1_list, acc5_list = np.zeros(args.num_encoders), np.zeros(args.num_encoders)

    iters_per_epoch = len(train_loader)
    
    kl_loss = torch.nn.KLDivLoss(reduction = "batchmean")
    for iter, data in enumerate(zip(train_loader, sup_train_loader)):
        images, labels = data[0][0], data[0][1]
        sup_images, sup_labels = data[1][0], data[1][1]
        data_time.update(time.time() - end)
        learning_rates.update(optimizer.param_groups[0]['lr'])
        if args.gpu is not None:
            # Size of images is args.batch_size * (args.num_augs + 1)
            images = images.reshape(-1, 3, datasets.img_size, datasets.img_size).cuda(args.gpu, non_blocking=True) 
            labels = labels.cuda(args.gpu, non_blocking=True)
            sup_images = sup_images.cuda(args.gpu, non_blocking=True)
            sup_labels = sup_labels.cuda(args.gpu, non_blocking=True)

        images = get_aug_wise_images(images, args)
        # sup_images, sup_labels = mixupcutmix(sup_images, sup_labels)

        with torch.cuda.amp.autocast(False):
            logits, feats = models_ensemble(images)
            # Last 'batch_size' number of images are unaugmented, there are (num_augs+1)*args.batch_size number of images in one batch
            # orig_image_logits, _ = models_ensemble(sup_images)
            orig_image_logits = logits[:, args.batch_size*args.num_augs:, :]

            # orig image logits and feats is of shape -  [args.num_encoders+1, .....], where last element corresponds to output of baseline model which is frozen
            ce_loss = get_loss(criterion, orig_image_logits[:-1], sup_labels)
            # Adding KL divergence loss between baseline and models_ensemble predictions
            kl_div = get_kl_loss(orig_image_logits[:-1], orig_image_logits[-1], kl_loss,  T=6, alpha=args.kl_coeff)

            similarity_matrix = get_similarity_vector(feats[:-1], args, running_mean)
            diff = get_pairwise_rowdiff(similarity_matrix).sum()
            loss =  (1 - kl_coeff) * ce_loss + coeff * diff + kl_div
            avg_sim += similarity_matrix.cpu().detach()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if model_ema and iter % args.model_ema_steps == 0:
            model_ema.update_parameters(models_ensemble)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)
        

        acc1, acc5 = [], []
        for i in range(0, args.num_encoders):
            a1, a5 = accuracy(orig_image_logits[i], sup_labels, topk=(1, 5))
            acc1.append(a1.item())
            acc5.append(a5.item())
            acc1_list[i] += a1.item()
            acc5_list[i] += a5.item()
        acc1 = np.mean(acc1)
        acc5 = np.mean(acc5)

        wandb.log({"acc": acc1, "ce loss": ce_loss, "diff": diff})
        ce_losses.update(ce_loss.item(), sup_images.size(0))
        variance.update(diff.item(), sup_images.size(0))
        kl.update(kl_div.item(), sup_images.size(0))
        top1.update(acc1, sup_images.size(0))
        top5.update(acc5, sup_images.size(0))
    
        batch_time.update(time.time() - end)
        end = time.time()
        # torch.cuda.empty_cache()
        if iter % args.print_freq == 0:
            progress.display(iter)

    acc1_list /= (iter+1)
    acc5_list /= (iter+1)
    avg_sim /= (iter+1)

    torch.cuda.empty_cache()
   
    return avg_sim, acc1_list, acc5_list, ce_losses.avg


def get_kl_loss(inputs, target, criterion, T, alpha):
    kl_div = 0.0
    for k in range(0, inputs.shape[0]):
        kl_div += criterion(F.log_softmax(inputs[k]/T, dim=1), F.softmax(target/T, dim=1)) * (alpha * T * T)
    return kl_div

def evaluate(val_loader, models_ensemble, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    ce_losses = AverageMeter('CE Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, learning_rates, ce_losses, top1, top5],
        prefix="Test: [{}]".format(epoch))

    models_ensemble.eval()
    end = time.time()
    avg_sim = 0.0

    acc1_list, acc5_list = np.zeros(args.num_encoders), np.zeros(args.num_encoders)
    orig_feats_list = []
    all_feats_list = [[] for _ in range(args.num_augs)]
    for iter, (images, labels) in enumerate(val_loader):
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.reshape(-1, 3, datasets.img_size, datasets.img_size).cuda(args.gpu, non_blocking=True) 
            labels = labels.cuda(args.gpu, non_blocking=True)
        
        images = get_aug_wise_images(images, args)
        with torch.no_grad():
            with torch.cuda.amp.autocast(False):
                logits, feats = models_ensemble(images) # Pass the un-augmented image
                # Last 'batch_size' images are unaugmented, there are (num_augs+1)*args.batch_size number of images in one batch
                orig_image_logits = logits[:, args.batch_size * args.num_augs:, :]
                orig_image_feats = feats[:, args.batch_size * args.num_augs:, :]
                orig_feats_list.append(orig_image_feats.cpu().detach())
                for n in range(args.num_augs):
                    all_feats_list[n].append(feats[:, n*args.batch_size:(n+1)*args.batch_size, :].cpu().detach())
                ce_loss = get_loss(criterion, orig_image_logits, labels)

            acc1, acc5 = [], []
            for i in range(0, args.num_encoders):
                a1, a5 = accuracy(orig_image_logits[i], labels, topk=(1, 5))
                acc1.append(a1.item())
                acc5.append(a5.item())
                acc1_list[i] += a1.item()
                acc5_list[i] += a5.item()
            acc1 = np.mean(acc1)
            acc5 = np.mean(acc5)

            ce_losses.update(ce_loss.item(), images.size(0)//(args.num_augs+1))
            top1.update(acc1, images.size(0)//(args.num_augs+1))
            top5.update(acc5, images.size(0)//(args.num_augs+1))

        batch_time.update(time.time() - end)
        end = time.time()
        # torch.cuda.empty_cache()
        if iter % args.print_freq == 0:
            progress.display(iter)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    acc1_list /= (iter+1)
    acc5_list /= (iter+1)
    all_orig_feats = torch.cat(orig_feats_list, dim = 1)
    sim = torch.zeros((args.num_encoders + 1, args.num_augs))
    
    for i in range(0, args.num_augs):
        all_aug_feats = all_feats_list[i]
        all_aug_feats = torch.cat(all_aug_feats, dim=1)
        sim[:, i] = F.cosine_similarity(all_orig_feats, all_aug_feats, dim = 2).mean(1)

    wandb.log({"val acc": top1.avg, "ce loss": ce_loss})
    return acc1_list, acc5_list, top1.avg, sim


def get_aug_wise_logits(logits, args):
    logits_list = []
    for i in range(args.num_augs+1):
        l = logits[:, i::args.num_augs+1, :]
        print(l.shape)
        torchvision.utils.save_image(l, "test_images_aug"+str(i)+ ".png")
        logits_list.append(logits[:, i::args.num_augs+1])
    logits_list = torch.cat(logits_list, dim = 1)
    return logits_list

def adjust_coeff(coeff, epochs, start_coeff, end_coeff = 5.0):
    coeff += (end_coeff - start_coeff)/epochs
    return coeff

def get_aug_wise_images(images, args):
    images_list = []
    for i in range(args.num_augs+1):
        l = images[i::args.num_augs+1, :, :, :]
        images_list.append(l)
    images_list = torch.cat(images_list, dim = 0)
    return images_list

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar') 

def get_loss(criterion, logits, labels):
    total_loss = 0.0
    for i in range(logits.shape[0]):
        loss = criterion(logits[i], labels)
        total_loss += loss
    total_loss /= logits.shape[0]
    return total_loss

def get_pairwise_rowdiff(sim_matrix, criterion = torch.nn.L1Loss()):
    diff = 0.0
    for i in range(0, sim_matrix.shape[0]):
        for j in range(i+1, sim_matrix.shape[0]):
            diff += torch.exp(-criterion(sim_matrix[i], sim_matrix[j]))
    return diff

def get_similarity_vector(feats, args, running_mean):
    # returns N vectors of R^k, each element being the similarity between original and augmented image
    sim = torch.zeros((args.num_encoders, args.num_augs)).cuda(args.gpu)

    # Unaugmented images 
    orig_feats = feats[:, args.batch_size * args.num_augs:, :]

    for i in range(0, args.num_augs):
        aug_feats = feats[:, args.batch_size * i: args.batch_size * (i+1), :]
        sim[:, i] = F.cosine_similarity(orig_feats, aug_feats, dim = 2).mean(1)
    return sim

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

