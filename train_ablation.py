import os
import sys
import time
import argparse
import logging
import numpy as np
import random

import torch
from torch import nn

from utils.data import datasets
from utils.model import models
from utils.evaluate import Evaluator
from utils.loss import myloss
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(seed=2022, epoches=80): #500
    parser = argparse.ArgumentParser(description='ablation')
    # dataset option
    parser.add_argument('--model_name', type=str, default='mosts', choices=['mosts'], help='model name')
    parser.add_argument('--data_loader', type=str, default='ablation_data_loader', choices=['ablation_data_loader'], help='data_loader name')
    parser.add_argument('--valid_group', type=int, default=3, help='set the valid group index (default: 0)')
    parser.add_argument('--train_batch_size', type=int, default=8, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='N', help='input batch size for testing (default: 16)')
    parser.add_argument('--num_workers', type=int, default=16, metavar='N', help='number of workers for data loader (default: 16)')
    parser.add_argument('--loss_name', type=str, default='combo', choices=['weighted_bce', 'dice', 'batch_dice', 'focal','combo','combo_batch', 'combo_mix'], help='set the loss function')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    rng_ = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    # Setup data generator
    mydataset_embedding = datasets[args.data_loader]
    data_train = mydataset_embedding(split='train', random_gen = rng_, num_candidates = 5, transform = None, transform_ref = None, valid_group=args.valid_group)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=args.train_batch_size, num_workers = args.num_workers, pin_memory=True, shuffle=True, prefetch_factor=2, worker_init_fn=worker_init_fn)
    data_val = mydataset_embedding(split='test', random_gen = rng_, num_candidates = 5, transform = None, transform_ref = None, valid_group=args.valid_group)
    loader_val = torch.utils.data.DataLoader(data_val, batch_size=args.test_batch_size, num_workers = args.num_workers, pin_memory=True, shuffle=False, prefetch_factor=2)
    
    evaluator = Evaluator(num_class=data_val.split_point+1) # ignore background class

    dir_name = 'log/' + str(args.data_loader) + '_' + str(args.model_name) + '_valid_group_' + str(args.valid_group)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    now_time = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    logging.basicConfig(level=logging.INFO,
                        filename=dir_name + '/output_' + now_time + '.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info('data_loader: %s, model_name: %s, loss_name: %s, batch_size: %s', args.data_loader, args.model_name, args.loss_name, args.train_batch_size)
    logging.info('train with: %s', data_train.train)
    logging.info('test with: %s', data_val.test)

    # Complie model
    model = models[args.model_name]()
    
    # CUDA init
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()

    # Setup loss function & optimizer, scheduler
    criterion = myloss[args.loss_name]()
    optim_para = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(optim_para, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(optim_para,lr=args.lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # Init loss & IoU
    IoU_final = 0
    epoch_final = 0
    losses = 0
    iteration = 0

    # Start training
    for epoch in range(epoches):
        train_loss = 0
        logging.info('epoch:' + str(epoch))
        start = time.time()
        np.random.seed(epoch)
        random.seed(epoch)
        data_train.curriculum = (epoch+1)/epoches
        data_train.random_gen = np.random.default_rng(epoch)

        for i, data in enumerate(loader_train):
            query, label, reference =  data[0], data[1], data[2]

            iteration += 1
            if torch.cuda.is_available():
                query = query.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                reference = reference.cuda(non_blocking=True)

            optimizer.zero_grad()
            output = model(query, reference)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            losses += loss.item()

            if iteration % 20 == 0:
                run_time = time.time() - start
                start = time.time()
                losses = losses / 20
                logging.info('iter:' + str(iteration) + " time:" + str(run_time) + " train loss = {:02.5f}".format(losses))
                losses = 0
        model_path = dir_name + '/epoch_{epoches}_texture.pth'.format(epoches=now_time)
        print("Training progress: ",data_train.curriculum*100,"%")

        # Model evaluation after one epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0
            evaluator.reset()
            np.random.seed(seed+1)
            random.seed(seed+1)
            data_val.curriculum = 1
            data_val.random_gen = np.random.default_rng(seed+1)

            for i, data in enumerate(loader_val):
                query, label, reference, image_class = data[0], data[1], data[2], data[3]

                if torch.cuda.is_available():
                    query = query.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                    reference = reference.cuda(non_blocking=True)

                scores = model(query, reference)
                val_loss += criterion(scores, label)
                seg = torch.clone(scores[:, 0, :, :].detach())
                seg[seg >= 0.5] = 1
                seg[seg < 0.5] = 0
 
                # Add batch sample into evaluator
                pred = seg.long().data.cpu().numpy()
                label = label.cpu().numpy()
                evaluator.add_batch(label, pred, image_class)

            mIoU, mIoU_d = evaluator.Mean_Intersection_over_Union()
            FBIoU = evaluator.FBIoU()

            logging.info("{:10s} {:.3f}".format('IoU_mean', mIoU))
            logging.info("{:10s} {}".format('IoU_mean_detail', mIoU_d))
            logging.info("{:10s} {:.3f}".format('FBIoU', FBIoU))
            if mIoU > IoU_final:
                epoch_final = epoch
                IoU_final = mIoU
                torch.save(model.state_dict(), model_path)
            logging.info('best_epoch:' + str(epoch_final))
            logging.info("{:10s} {:.3f}".format('best_IoU', IoU_final))
        
        model.train()
        scheduler.step()
        logging.info(f"LR: {optimizer.param_groups[0]['lr']}")

    logging.info(epoch_final)
    logging.info(IoU_final)

if __name__ == '__main__':
    main()