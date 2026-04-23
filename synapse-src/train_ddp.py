import os
import time
from time import ctime

# from sklearnex import patch_sklearn
# patch_sklearn()

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from importlib import import_module
import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from torcheval.metrics.aggregation.auc import AUC
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score

import shutil
import warnings
import builtins
import random
import math
import sys
from shutil import copyfile

from dataset import ImageDataset, collate_fn
from model.utils import get_model
from utils import AverageMeter, calculate_accuracy, save_checkpoint, adjust_learning_rate, ProgressMeter

from config import create_arg_parser


def main_worker(gpu, ngpus_per_node, args, save_dir):
    from path import CHECKPOINT_DIR, TENSORBOARD_DIR, LOG_DIR
    from logger import setup_logger
    
    mod_name = f"{args.ablation_mode}_{args.integration_method}"
    
    # Clear GPU cache from any previous runs
    torch.cuda.empty_cache()
    
    # Initialize loggers for GPU 0 (or master process)
    if not args.multiprocessing_distributed or gpu == 0:
        # Create modality specific log directory
        modal_log_dir = os.path.join(LOG_DIR, mod_name)
        os.makedirs(modal_log_dir, exist_ok=True)
        logger = setup_logger(save_dir=modal_log_dir, name=f"train_{mod_name}")
        test_logger = setup_logger(save_dir=modal_log_dir, name="test")
        builtins.print = logger.info
        
    writer = SummaryWriter(log_dir=save_dir.replace(CHECKPOINT_DIR, TENSORBOARD_DIR))

    args.gpu = gpu

    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        mod_name = f"{args.ablation_mode}_{args.integration_method}"
        print("Use GPU: {} for training combination: {}.".format(args.gpu, mod_name))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            print("gpu... ", gpu)
            args.rank = args.rank * ngpus_per_node + gpu
            # args.rank = gpu
        dist.init_process_group(
            backend = args.dist_backend,
            init_method=f"{args.dist_url}",
            world_size = args.world_size,
            rank = args.rank
        )

    # model = get_model(args, weights=args.pretrained_weights)
    model = get_model(args)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)

            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

            # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            generator = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            generator = DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        generator = model.cuda(args.gpu)
    else:
        generator = model.cuda()
    
    
    cudnn.benchmark = True

    # criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion = torch.nn.BCEWithLogitsLoss().cuda(args.gpu)
    criterion_CLIP = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    if args.num_classes > 2:
        criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = torch.nn.BCELoss().cuda(args.gpu)
    
    # if 'textCosSim' in args.loss:
    criterion_CosSim = torch.nn.CosineEmbeddingLoss().cuda(args.gpu)
    
    if args.learnablePrompt:
        args.lr = 0.001
        optimizer = torch.optim.SGD(generator.parameters(),
                                    lr=args.lr,
                                    weight_decay=10**-7
                                    )
    else:
        if args.num_classes > 2:
            args.lr = 0.001
        else:
            args.lr = 0.00001
        optimizer = torch.optim.Adam(generator.parameters(),
                                    lr=args.lr,
                                    betas=(args.b1, args.b2),
                                    weight_decay=10**-7)
    
    if args.resume:
        if args.resume[:4] != '/mnt':
            args.resume = '/mnt/KW/LungCancer/Multimodality2/results/SavedModels/' + args.resume + '/checkpoint_best.pt'
                
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            if args.pretrainedExt_freeze:
                if args.pretrainedExt_CT and 'CT' in args.modality:
                    try:
                        for p in model.extractor_CT.parameters():
                            p.requires_grad = False
                    except Exception: pass
                if args.pretrainedExt_pathology and 'pathology' in args.modality:
                    try:
                        for p in model.extractor_pathology.parameters():
                            p.requires_grad = False
                    except Exception: pass
                if args.pretrainedExt_CI and 'CI' in args.modality:
                    try:
                        for p in model.extractor_CI.parameters():
                            p.requires_grad = False
                    except Exception: pass
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.pretrainedExt_CT and 'CT' in args.modality:
            try:
                pretrained_CT_extractor = torch.load(args.pretrainedExt_CT_pth)
                pretrained_CT_extractor_dict = pretrained_CT_extractor["state_dict"]
                model_CT_dict = model.extractor_CT.state_dict()
                pretrained_CT_extractor_dict = {k: v for k, v in pretrained_CT_extractor_dict.items() if k in model_CT_dict}
                model_CT_dict.update(pretrained_CT_extractor_dict)
                model.extractor_CT.load_state_dict(model_CT_dict)
                if args.pretrainedExt_freeze:
                    for p in model.extractor_CT.parameters():
                        p.requires_grad = False
            except Exception: pass
        
        if args.pretrainedExt_pathology and 'pathology' in args.modality:
            try:
                pretrained_pathology_extractor = torch.load(args.pretrainedExt_pathology_pth)
                pretrained_pathology_extractor_dict = pretrained_pathology_extractor["state_dict"]
                model_pathology_dict = model.extractor_pathology.state_dict()
                pretrained_pathology_extractor_dict = {k: v for k, v in pretrained_pathology_extractor_dict.items() if k in model_pathology_dict}
                model_pathology_dict.update(pretrained_pathology_extractor_dict)
                model.extractor_pathology.load_state_dict(model_pathology_dict)
                if args.pretrainedExt_freeze:
                    for p in model.extractor_pathology.parameters():
                        p.requires_grad = False
            except Exception: pass
        
        if args.pretrainedExt_CI and 'CI' in args.modality:
            try:
                pretrained_CI_extractor = torch.load(args.pretrainedExt_CI_pth)
                pretrained_CI_extractor_dict = pretrained_CI_extractor["state_dict"]
                model_CI_dict = model.extractor_CI.state_dict()
                pretrained_CI_extractor_dict = {k: v for k, v in pretrained_CI_extractor_dict.items() if k in model_CI_dict}
                model_CI_dict.update(pretrained_CI_extractor_dict)
                model.extractor_CI.load_state_dict(model_CI_dict)
                if args.pretrainedExt_freeze:
                    for p in model.extractor_CI.parameters():
                        p.requires_grad = False
            except Exception: pass
    

    train_dataset = ImageDataset(args, mode='train')
    valid_dataset = ImageDataset(args, mode='valid')
    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    else:
        train_sampler = None

    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)
    dataloader_valid = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,                   num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    
    valid_auc_best = 0
    for epoch in range(args.start_epoch, args.n_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        
        # print(save_dir)
        print("---------------------------------------------------------------------------------------------------------------------------")
        train(dataloader_train, generator, criterion, criterion_CosSim, optimizer, epoch, args, writer)
        torch.cuda.empty_cache()  # Free training memory before validation
        print("---------------------------------------------------------------------------------------------------------------------------")
        _, valid_acc, valid_auc, valid_bacc, valid_f1 = valid(dataloader_valid, generator, criterion, criterion_CosSim, optimizer, epoch, args, writer)
        torch.cuda.empty_cache()  # Free validation memory
        if not args.multiprocessing_distributed or args.rank % ngpus_per_node == 0:
            test_logger.info(f"[{mod_name}] Epoch {epoch} | AUC: {valid_auc:.4f} | BACC: {valid_bacc:.4f} | F1: {valid_f1:.4f}")
            # Log for final evaluation summary
            if epoch == args.n_epochs - 1:
                 modal_log_dir = os.path.join(LOG_DIR, mod_name)
                 with open(os.path.join(modal_log_dir, "final_test_evaluation.log"), "a") as f:
                     f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {mod_name} | Epoch: {epoch} | AUC: {valid_auc:.4f} | BACC: {valid_bacc:.4f} | F1: {valid_f1:.4f} | Acc: {valid_acc:.4f}\n")
        print("---------------------------------------------------------------------------------------------------------------------------")

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            is_best = args.save_best
            if is_best:
                if valid_auc_best <= valid_auc:
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        is_best=is_best,
                        save_dir = save_dir,
                        filename="checkpoint_{:04d}.pt".format(epoch),
                    )
                    valid_auc_best = valid_auc
            else:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=is_best,
                    save_dir = save_dir,
                    filename="checkpoint_{:04d}.pt".format(epoch),
                )
            torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, save_dir + '/checkpoint_last.pt')

def train(dataloader_train, generator, criterion, criterion_CosSim, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    losses_CosSim = AverageMeter("Loss", ":.4e")
    if args.loss_point == 'CT-Pth-Last':
        losses_CT = AverageMeter("Loss_CT", ":.4e")
        losses_Pth = AverageMeter("Loss_Pth", ":.4e")
        losses_Last = AverageMeter("Loss_Last", ":.4e")
    accs = AverageMeter("Acc", ":6.2f")
    from utils import SummaryMeter
    mod_name = f"{args.ablation_mode}_{args.integration_method}"
    
    aucs = SummaryMeter("AUC", ":.4f")
    f1s = SummaryMeter("F1", ":.4f")
    baccs = SummaryMeter("BACC", ":.4f")
    
    progress = ProgressMeter(
        len(dataloader_train),
        [batch_time, data_time, losses, accs, aucs, f1s, baccs],
        prefix="[{}] Train Epoch: [{}]".format(mod_name, epoch),
    )
    
    # AUC_metric = AUC()
    preds_scores = []
    preds_labels = []
    labels = []
    
    label_for_CosSim = torch.tensor([1], dtype=torch.float32).cuda(args.gpu)

    generator.train()

    end = time.time()
    for i, train_data_dict in enumerate(dataloader_train):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            train_input_CT = train_data_dict['input_CT'].cuda(non_blocking=True)
            train_input_pathology = train_data_dict['input_pathology'].cuda(non_blocking=True)
            train_input_CI = train_data_dict['input_CI'].cuda(non_blocking=True)
                
            if 'wMask' in args.model_CT:
                train_mask = train_data_dict['mask'].cuda(non_blocking=True)
            
            if 'pathology' in args.modality and args.model_pathology == 'ABMIL_v2':
                train_info_BpRc = train_data_dict.get('BpRc_class', torch.tensor(0)).float().cuda(non_blocking=True)
            
            train_label = train_data_dict['label'].float().cuda(non_blocking=True)

        if ('CT' in args.modality) & ('pathology' in args.modality):
            if 'wMask' in args.model_CT:
                train_output, train_CT2CI, train_Pth2CI, _ = generator([train_input_CT, train_input_pathology], train_input_CI, train_mask)
            else:
                train_output, train_CT2CI, train_Pth2CI = generator([train_input_CT, train_input_pathology], train_input_CI)
        elif ('CT' in args.modality):
            if args.alignment_base == 'none':
                train_output, _ = generator([train_input_CT], train_input_CI)
            else:
                if 'wMask' in args.model_CT:
                    train_output, train_CT2CI, _ = generator([train_input_CT], train_input_CI, train_mask)
                else:
                    train_output, train_CT2CI, _ = generator([train_input_CT], train_input_CI)
        elif ('pathology' in args.modality):
            if args.model_pathology == 'ABMIL_v2':
                train_output, train_Pth2CI = generator([train_input_pathology, train_info_BpRc], train_input_CI)
            else:
                train_output, train_Pth2CI = generator([train_input_pathology], train_input_CI)
        elif ('CI' in args.modality):
            train_output = generator([], train_input_CI)
        
        if args.loss_point == 'CT-Pth-Last':
            train_loss_CT = criterion(train_output_CTonly, train_label)
            train_loss_Pth = criterion(train_output_Pthonly, train_label)
            train_loss_Last = criterion(train_output, train_label)
            train_loss = train_loss_CT + train_loss_Pth + train_loss_Last
        elif args.loss_point == 'Last':
            train_loss = criterion(train_output, train_label)
        if 'textCosSim' in args.loss:
            train_loss_textCosSim = criterion_CosSim(train_CT2CI.squeeze(1), train_Pth2CI.squeeze(1), label_for_CosSim)
            losses_CosSim.update(train_loss_textCosSim.item(), train_output.size(0))
            
            train_loss += train_loss_textCosSim
        
        train_acc = calculate_accuracy(train_output, train_label)

        losses.update(train_loss.item(), train_output.size(0))
        accs.update(train_acc.item(), train_output.size(0))
        if args.loss_point == 'CT-Pth-Last':
            losses_CT.update(train_loss_CT.item(), train_output.size(0))
            losses_Pth.update(train_loss_Pth.item(), train_output.size(0))
            losses_Last.update(train_loss_Last.item(), train_output.size(0))
        # AUC_metric.update(train_output[:,0], train_label)
        
        for b in range(train_output.shape[0]):
            # preds.append(train_output[b,0].item())
            preds_scores.append(train_output[b, 1].item())
            preds_labels.append(torch.argmax(train_output[b,:]).item())
            labels.append(torch.argmax(train_label[b,:]).item())

        # Update running metrics
        if len(set(labels)) > 1:
            try:
                if args.num_classes > 2:
                    current_auc = roc_auc_score(np.eye(args.num_classes)[labels], np.eye(args.num_classes)[preds_labels], multi_class='ovo', average='macro')
                else:
                    current_auc = roc_auc_score(labels, preds_scores)
                aucs.update(current_auc)
            except:
                pass
        
        current_f1 = f1_score(labels, preds_labels, average='macro' if args.num_classes > 2 else 'binary', zero_division=0)
        current_bacc = balanced_accuracy_score(labels, preds_labels)
        f1s.update(current_f1)
        baccs.update(current_bacc)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        if i % 10 == 0:
            progress.display(i)
    
    
    writer.add_scalar('train/loss', losses.avg, epoch)
    if args.loss_point == 'CT-Pth-Last':
        writer.add_scalar('train/loss_CT', losses_CT.avg, epoch)
        writer.add_scalar('train/loss_Pth', losses_Pth.avg, epoch)
        writer.add_scalar('train/loss_Last', losses_Last.avg, epoch)
    writer.add_scalar('train/loss_CosSim', losses_CosSim.avg, epoch)
    writer.add_scalar('train/acc', accs.avg, epoch)
    if args.num_classes > 2:
        train_auc = roc_auc_score(np.eye(args.num_classes)[labels], np.eye(args.num_classes)[preds_labels], multi_class='ovo', average='macro')
    else:
        train_auc = roc_auc_score(labels, preds_scores)
    writer.add_scalar('train/auc', train_auc, epoch)
    
    preds_binary = [round(item) for item in preds_labels]
    if args.num_classes > 2:
        train_recall = recall_score(labels, preds_binary, average='macro', zero_division=np.nan)
        train_precision = precision_score(labels, preds_binary, average='macro', zero_division=np.nan)
        train_f1 = f1_score(labels, preds_binary, average='macro', zero_division=np.nan)
        train_bacc = balanced_accuracy_score(labels, preds_binary)
    else:
        train_recall = recall_score(labels, preds_binary, zero_division=np.nan)
        train_precision = precision_score(labels, preds_binary, zero_division=np.nan)
        train_f1 = f1_score(labels, preds_binary, zero_division=np.nan)
        train_bacc = balanced_accuracy_score(labels, preds_binary)
    writer.add_scalar('train/recall', train_recall, epoch)
    writer.add_scalar('train/precision', train_precision, epoch)
    writer.add_scalar('train/f1', train_f1, epoch)
    writer.add_scalar('train/bacc', train_bacc, epoch)


def valid(dataloader_valid, generator, criterion, criterion_CosSim, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    losses_CosSim = AverageMeter("Loss", ":.4e")
    if args.loss_point == 'CT-Pth-Last':
        losses_CT = AverageMeter("Loss_CT", ":.4e")
        losses_Pth = AverageMeter("Loss_Pth", ":.4e")
        losses_Last = AverageMeter("Loss_Last", ":.4e")
    accs = AverageMeter("Acc", ":6.2f")
    from utils import SummaryMeter
    mod_name = f"{args.ablation_mode}_{args.integration_method}"
    
    aucs = SummaryMeter("AUC", ":.4f")
    f1s = SummaryMeter("F1", ":.4f")
    baccs = SummaryMeter("BACC", ":.4f")
    
    progress = ProgressMeter(
        len(dataloader_valid),
        [batch_time, data_time, losses, accs, aucs, f1s, baccs],
        prefix="[{}] Valid Epoch: [{}]".format(mod_name, epoch),
    )
    
    # AUC_metric = AUC()
    preds_scores = []
    preds_labels = []
    labels = []
    
    label_for_CosSim = torch.tensor([1], dtype=torch.float32).cuda(args.gpu)
    
    with torch.no_grad():
        generator.eval()

        end = time.time()
        for i, valid_data_dict in enumerate(dataloader_valid):
            data_time.update(time.time() - end)

            if args.gpu is not None:
                valid_input_CT = valid_data_dict['input_CT'].cuda(non_blocking=True)
                valid_input_pathology = valid_data_dict['input_pathology'].cuda(non_blocking=True)
                valid_input_CI = valid_data_dict['input_CI'].cuda(non_blocking=True)
                    
                if 'wMask' in args.model_CT:
                     valid_mask = valid_data_dict['mask'].cuda(non_blocking=True)
            
                if 'pathology' in args.modality and args.model_pathology == 'ABMIL_v2':
                    valid_info_BpRc = valid_data_dict.get('BpRc_class', torch.tensor(0)).float().cuda(non_blocking=True)
                
                valid_label = valid_data_dict['label'].float().cuda(non_blocking=True)

            if ('CT' in args.modality) & ('pathology' in args.modality):
                valid_output, valid_CT2CI, valid_Pth2CI = generator([valid_input_CT, valid_input_pathology], valid_input_CI)
            elif ('CT' in args.modality):
                if args.alignment_base == 'none':
                    valid_output, _ = generator([valid_input_CT], valid_input_CI)
                else:
                    if 'wMask' in args.model_CT:
                        valid_output, valid_CT2CI, _ = generator([valid_input_CT], valid_input_CI, valid_mask)
                    else:
                        valid_output, valid_CT2CI, _ = generator([valid_input_CT], valid_input_CI)
            elif ('pathology' in args.modality):
                if args.model_pathology == 'ABMIL_v2':
                    valid_output, valid_Pth2CI = generator([valid_input_pathology, valid_info_BpRc], valid_input_CI)
                else:
                    valid_output, valid_Pth2CI = generator([valid_input_pathology], valid_input_CI)
            elif ('CI' in args.modality):
                valid_output = generator([], valid_input_CI)
            
            if args.loss_point == 'CT-Pth-Last':
                valid_loss_CT = criterion(valid_output_CTonly, valid_label)
                valid_loss_Pth = criterion(valid_output_Pthonly, valid_label)
                valid_loss_Last = criterion(valid_output, valid_label)
                valid_loss = valid_loss_CT + valid_loss_Pth + valid_loss_Last
            elif args.loss_point == 'Last':
                valid_loss = criterion(valid_output, valid_label)
            if 'textCosSim' in args.loss:
                valid_loss_textCosSim = criterion_CosSim(valid_CT2CI.squeeze(1), valid_Pth2CI.squeeze(1), label_for_CosSim)
                losses_CosSim.update(valid_loss_textCosSim.item(), valid_output.size(0))
                
                valid_loss += valid_loss_textCosSim
            
            valid_acc = calculate_accuracy(valid_output, valid_label)

            losses.update(valid_loss.item(), valid_output.size(0))
            accs.update(valid_acc.item(), valid_output.size(0))
            if args.loss_point == 'CT-Pth-Last':
                losses_CT.update(valid_loss_CT.item(), valid_output.size(0))
                losses_Pth.update(valid_loss_Pth.item(), valid_output.size(0))
                losses_Last.update(valid_loss_Last.item(), valid_output.size(0))
            # AUC_metric.update(valid_output[:,0], valid_label)
        
            for b in range(valid_output.shape[0]):
                # preds.append(valid_output[b,0].item())
                preds_scores.append(valid_output[b,1].item())
                preds_labels.append(torch.argmax(valid_output[b,:]).item())
                labels.append(torch.argmax(valid_label[b,:]).item())

                # Update running metrics
            if len(set(labels)) > 1:
                try:
                    if args.num_classes > 2:
                        current_auc = roc_auc_score(np.eye(args.num_classes)[labels], np.eye(args.num_classes)[preds_labels], multi_class='ovo', average='macro')
                    else:
                        current_auc = roc_auc_score(labels, preds_scores)
                    aucs.update(current_auc)
                except:
                    pass
            
            current_f1 = f1_score(labels, preds_labels, average='macro' if args.num_classes > 2 else 'binary', zero_division=0)
            current_bacc = balanced_accuracy_score(labels, preds_labels)
            f1s.update(current_f1)
            baccs.update(current_bacc)

            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            if i % 10 == 0:
                progress.display(i)
    
    
    writer.add_scalar('valid/loss', losses.avg, epoch)
    if args.loss_point == 'CT-Pth-Last':
        writer.add_scalar('valid/loss_CT', losses_CT.avg, epoch)
        writer.add_scalar('valid/loss_Pth', losses_Pth.avg, epoch)
        writer.add_scalar('valid/loss_Last', losses_Last.avg, epoch)
    writer.add_scalar('valid/loss_CosSim', losses_CosSim.avg, epoch)
    writer.add_scalar('valid/acc', accs.avg, epoch)
    if args.num_classes > 2:
        valid_auc = roc_auc_score(np.eye(args.num_classes)[labels], np.eye(args.num_classes)[preds_labels], multi_class='ovo', average='macro')
    else:
        valid_auc = roc_auc_score(labels, preds_scores)
    writer.add_scalar('valid/auc', valid_auc, epoch)
    
    preds_binary = [round(item) for item in preds_labels]
    if args.num_classes > 2:
        valid_recall = recall_score(labels, preds_binary, average='macro', zero_division=np.nan)
        valid_precision = precision_score(labels, preds_binary, average='macro', zero_division=np.nan)
        valid_f1 = f1_score(labels, preds_binary, average='macro', zero_division=np.nan)
        valid_bacc = balanced_accuracy_score(labels, preds_binary)
    else:
        valid_recall = recall_score(labels, preds_binary, zero_division=np.nan)
        valid_precision = precision_score(labels, preds_binary, zero_division=np.nan)
        valid_f1 = f1_score(labels, preds_binary, zero_division=np.nan)
        valid_bacc = balanced_accuracy_score(labels, preds_binary)
    writer.add_scalar('valid/recall', valid_recall, epoch)
    writer.add_scalar('valid/precision', valid_precision, epoch)
    writer.add_scalar('valid/f1', valid_f1, epoch)
    writer.add_scalar('valid/bacc', valid_bacc, epoch)

    return losses.avg, accs.avg, valid_auc, valid_bacc, valid_f1



if __name__ == "__main__":
    args = create_arg_parser()
    
    testHospitalName = args.hospital_test[0]
    if len(args.hospital_test) > 1:
        for i in range(len(args.hospital_test)-1):
            i += 1
            testHospitalName += '+' + args.hospital_test[i]
    
    if args.tumorCrop:
        checkTumorCrop = 'O'
    else:
        checkTumorCrop = 'X'
    
    if 'wMask' in args.model_CT:
        maskOX = 'O'
    else:
        maskOX = 'X'
    
    args.modality = []
    if 'CT' in args.ablation_mode: args.modality.append('CT')
    if 'WSI' in args.ablation_mode: args.modality.append('pathology')
    if 'Clinical' in args.ablation_mode: args.modality.append('CI')

    mod_name = f"{args.ablation_mode}_{args.integration_method}"
    
    train_start_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
    model_name = "-".join([args.model_CT if m=='CT' else args.model_pathology if m=='pathology' else args.model_CI for m in args.modality])
    model_name += f"({args.aggregator})"

    if 'CT' in args.modality:
        save_dir = '%s/%s/stage_tr(%s)/%s/norm_[%s]/mask(%s)/crop(%s)/[%d]%s' % (mod_name,
                                                                                       testHospitalName,
                                                                                       args.cancerstageTrain,
                                                                                       model_name,
                                                                                       str(args.spacing[0])+','+str(args.spacing[1])+','+str(args.spacing[2]),
                                                                                       maskOX,
                                                                                       checkTumorCrop,
                                                                                       args.val_fold,
                                                                                       train_start_time)
    else:
        save_dir = '%s/%s/stage_tr(%s)/%s/norm_[%s]/[%d]%s'                 % (mod_name,
                                                                                  testHospitalName,
                                                                                  args.cancerstageTrain,
                                                                                  model_name,
                                                                                  str(args.spacing[0])+','+str(args.spacing[1])+','+str(args.spacing[2]),
                                                                                  args.val_fold,
                                                                                  train_start_time)
    from path import CHECKPOINT_DIR
    save_dir = os.path.join(CHECKPOINT_DIR, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"FINAL_SAVE_DIR: {save_dir}")
    
    with open(f"{save_dir}/config.txt", 'w') as f:
        for key in vars(args).keys():
            f.write(f"{key}: {vars(args)[key]}\n")
    copyfile('model/aggregator.py', save_dir+'/aggregator.py')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # os.environ['MASTER_ADDR'] = args.master_IP
    # os.environ['MASTER_PORT'] = args.master_port
    
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    cudnn.enabled = True
    
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # os.environ["PYTHONHASHSEED"] = str(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, save_dir))
    else:
        main_worker(0 if args.gpu is not None else None, ngpus_per_node, args, save_dir)


    sys.exit(0)