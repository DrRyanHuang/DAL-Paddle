from __future__ import print_function

import os
import argparse
import numpy as np
import time
import glob
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.distributed as dist
from paddle.distributed import fleet, get_rank
from paddle.io import DistributedBatchSampler

import matplotlib.pyplot as plt
from tqdm import tqdm

from models.model import RetinaNet
from eval import evaluate
from datasets import *
from utils.utils import *
# from torch_warmup_lr import WarmupLR


DATASETS = {'VOC' : VOCDataset ,
            'IC15': IC15Dataset,
            'IC13': IC13Dataset,
            'HRSC2016': HRSCDataset,
            'DOTA':DOTADataset,
            'UCAS_AOD':UCAS_AODDataset,
            'NWPU_VHR':NWPUDataset
            }


def train_model(args, hyps):
    #  parse configs
    epochs = int(hyps['epochs'])
    batch_size = int(hyps['batch_size'])
    results_file = 'result.txt'
    weight =  'weights' + os.sep + 'last.pth' if args.resume or args.load else args.weight
    last = 'weights' + os.sep + 'last.pth'
    best = 'weights' + os.sep + 'best.pth'
    start_epoch = 0
    best_fitness = 0 #   max f1
    # device = paddle.device("cuda:0" if paddle.cuda.is_available() else "cpu")

    # creat folder
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    for f in glob.glob(results_file):
        # os.remove(f)
        pass

    # multi-scale
    if args.multi_scale:
        scales = args.training_size + 32 * np.array([x for x in range(-1, 5)])
        # set manually
        # scales = np.array([384, 480, 544, 608, 704, 800, 896, 960])
        print('Using multi-scale %g - %g' % (scales[0], scales[-1]))   
    else :
        scales = args.training_size 


    # dataloader
    assert args.dataset in DATASETS.keys(), 'Not supported dataset!'
    ds = DATASETS[args.dataset](dataset=args.train_path, augment=args.augment)
    collater = Collater(scales=scales, keep_ratio=True, multiple=32)
    
    if not args.fleet:
        loader = paddle.io.DataLoader(
            dataset=ds,
            batch_size=batch_size,
            num_workers=8,
            collate_fn=collater,
            shuffle=True,
            use_shared_memory=False,
            drop_last=True
        )
    else:
        sampler = DistributedBatchSampler(ds,
                                  rank=get_rank(),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,)
        loader = paddle.io.DataLoader(
            dataset=ds,
            batch_sampler=sampler,
            num_workers=8,
            collate_fn=collater,
            use_shared_memory=False,
        )
    # Initialize model
    init_seeds()
    model = RetinaNet(backbone=args.backbone, hyps=hyps)

    grad_clip=paddle.nn.ClipGradByGlobalNorm(0.1)
    
    # Optimizer
    # scheduler = optim.lr.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    scheduler = optim.lr.MultiStepDecay(hyps['lr0'], milestones=[round(epochs * x) for x in [0.7, 0.9]], gamma=0.1)
    optimizer = optim.Adam(parameters=model.parameters(), learning_rate=scheduler, grad_clip=grad_clip)
    # scheduler = WarmupLR(scheduler, init_lr=hyps['warmup_lr'], num_warmup=hyps['warm_epoch'], warmup_strategy='cos')
    # scheduler = paddle.optim.lr.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min = 1e-5)
    scheduler.last_epoch = start_epoch - 1
    ######## Plot lr schedule #####
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, label='LR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)    
    # import ipdb; ipdb.set_trace()
    ###########################################

    # load chkpt
    if weight.endswith('.pth'):
        chkpt = paddle.load(weight)
        # load model
        if 'model' in chkpt.keys() :
            model.set_state_dict(chkpt['model'])
        else:
            model.set_state_dict(chkpt)
        # load optimizer
        if 'optimizer' in chkpt.keys() and chkpt['optimizer'] is not None and args.resume :
            optimizer.set_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']
            
            # for state in optimizer.state.values():
            #     for k, v in state.items():
            #         if isinstance(v, paddle.Tensor):
            #             state[k] = v.cuda()
            
        # load results
        if 'training_results' in chkpt.keys() and  chkpt.get('training_results') is not None and args.resume:
            with open(results_file, 'a') as file:
                file.write(chkpt['training_results'])  # write results.txt
        if args.resume and 'epoch' in chkpt.keys():
            start_epoch = chkpt['epoch'] + 1   

        del chkpt
    
    model_info(model, report='summary')  # 'full' or 'summary'
    
    if args.fleet:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
   

    if args.amp:
        amp_scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
        model, optimizer = paddle.amp.decorate(models=model, optimizers=optimizer, level=args.amp_level)

    
    # 'P', 'R', 'mAP', 'F1'
    results = (0, 0, 0, 0)

    for epoch in range(start_epoch,epochs):
        print(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem',  'cls', 'reg', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(loader), total=len(loader))  # progress bar
        mloss = np.zeros([2])
        for i, (ni, batch) in enumerate(pbar):
            
            model.train()

            if args.freeze_bn:
                if paddle.device.cuda.device_count() > 1:
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()

            optimizer.clear_grad()
            ims, gt_boxes = batch['image'], batch['boxes']
            with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level=args.amp_level, enable=args.amp):
                losses = model(ims, gt_boxes, process=epoch/epochs )
            loss_cls, loss_reg = losses['loss_cls'].mean(), losses['loss_reg'].mean()
            loss = loss_cls + loss_reg
            if not paddle.isfinite(loss):
                import ipdb; ipdb.set_trace()
                print('WARNING: non-finite loss, ending training ')
                break
            if bool(loss == 0):
                continue

            # calculate gradient
            if args.amp:
                scaled_loss = amp_scaler.scale(loss)
                scaled_loss.backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Print batch results
            loss_items = np.array([loss_cls.detach().numpy().item(), loss_reg.detach().numpy().item()])
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = paddle.device.cuda.max_memory_reserved() / 1E9 if True else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 5) % (
                  '%g/%g' % (epoch, epochs - 1), 
                  '%.3gG' % mem, 
                  *mloss, mloss.sum().item(), gt_boxes.shape[1], min(ims.shape[2:]))
            pbar.set_description(s)

        # Update scheduler
        scheduler.step()
        final_epoch = epoch + 1 == epochs
        
        # eval
        if hyps['test_interval']!= -1 and epoch % hyps['test_interval'] == 0 and epoch > 30 :
            if paddle.device.cuda.device_count() > 1:
                results = evaluate(target_size=args.target_size,
                                   test_path=args.test_path,
                                   dataset=args.dataset,
                                   model=model.module, 
                                   hyps=hyps,
                                   conf = 0.01 if final_epoch else 0.1)    
            else:
                results = evaluate(target_size=args.target_size,
                                   test_path=args.test_path,
                                   dataset=args.dataset,
                                   model=model,
                                   hyps=hyps,
                                   conf = 0.01 if final_epoch else 0.1) #  p, r, map, f1

        
        # Write result log
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 4 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        ##   Checkpoint
        if arg.dataset in ['IC15', ['IC13']]:
            fitness = results[-1]   # Update best f1
        else :
            fitness = results[-2]   # Update best mAP
        if fitness > best_fitness:
            best_fitness = fitness

        with open(results_file, 'r') as f:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_fitness': best_fitness,
                     'training_results': f.read(),
                     'model': model.state_dict(),
                     'optimizer': None if final_epoch else optimizer.state_dict()}
        

        # Save last checkpoint
        paddle.save(chkpt, last)
        # Save best checkpoint
        if best_fitness == fitness:
            paddle.save(chkpt, best) 

        if (epoch % hyps['save_interval'] == 0  and epoch > 100) or final_epoch:
            if paddle.device.cuda.device_count() > 1:
                paddle.save(chkpt, './weights/deploy%g.pth'% epoch)
            else:
                paddle.save(chkpt, './weights/deploy%g.pth'% epoch)

    # end training
    dist.destroy_process_group() if paddle.device.cuda.device_count() > 1 else None
    paddle.device.cuda.empty_cache()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a detector')
    # config
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    # network
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--freeze_bn', type=bool, default=False)
    parser.add_argument('--weight', type=str, default='')   # 
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')

    # NWPU-VHR10
    parser.add_argument('--dataset', type=str, default='NWPU_VHR')
    parser.add_argument('--train_path', type=str, default='NWPU_VHR/train.txt')
    parser.add_argument('--test_path', type=str, default='NWPU_VHR/test.txt')

    parser.add_argument('--training_size', type=int, default=800)
    parser.add_argument('--resume', action='store_true', help='resume training from last.pth')
    parser.add_argument('--load', action='store_true', help='load training from last.pth')
    parser.add_argument('--augment', action='store_true', help='data augment')
    parser.add_argument('--target_size', type=int, default=[800])   
    #
    
    parser.add_argument('--fleet', action='store_true', default=False, help='whether to use fleet')
    parser.add_argument('--amp', action='store_true', default=False, help='whether to use amp')
    parser.add_argument('--amp_level', type=str, default='O1', help='amp level O1 or O2')

    arg = parser.parse_args()
    hyps = hyp_parse(arg.hyp)
    print(arg)
    print(hyps)
    
    if arg.fleet:
        fleet.init()

    train_model(arg, hyps)