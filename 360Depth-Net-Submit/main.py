from __future__ import print_function
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from tensorboardX import SummaryWriter
from tqdm import tqdm
import utils_func
from models import LCV_ours_sub3

from torch.optim.lr_scheduler import MultiStepLR, StepLR
import logger
import shutil

parser = argparse.ArgumentParser(description='360SD-Net')

parser.add_argument('--save_path', type=str, default='', help='path to save log and checkpoints')
parser.add_argument('--maxdisp', type=int, default=68, help='maxium disparity')
parser.add_argument('--maxdepth', type=int, default=80,
                    help='the range of the depth cost volume is from to max depth')
parser.add_argument('--model', default='360SDNet', help='select model')
parser.add_argument('--datapath', default='data/', help='datapath')
parser.add_argument('--datapath2', default='data/', help='datapath2')
parser.add_argument('--datapath_val',
                    default='data/MP3D/val/',
                    help='datapath for validation')
parser.add_argument('--epochs',
                    type=int,
                    default=500,
                    help='number of epochs to train')
parser.add_argument('--start_decay',
                    type=int,
                    default=400,
                    help='number of epoch for lr to start decay')
parser.add_argument('--start_learn',
                    type=int,
                    default=50,
                    help='number of epoch for LCV to start learn')
parser.add_argument('--batch',
                    type=int,
                    default=16,
                    help='number of batch to train')
parser.add_argument('--checkpoint', default=None, help='load checkpoint path')
parser.add_argument('--save_checkpoint',
                    default='./checkpoints',
                    help='save checkpoint path')
parser.add_argument('--tensorboard_path',
                    default='./logs',
                    help='tensorboard path')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
parser.add_argument('--real',
                    action='store_true',
                    default=False,
                    help='adapt to real world images')
parser.add_argument('--SF3D',
                    action='store_true',
                    default=False,
                    help='read stanford3D data')
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--w_p', type=int, default=1,
                    help='Corresponds to p-Wasserstein distance')
parser.add_argument("--scale", type=int, default=1, help="reduce x times resolution for the grids")
# ------------------------ learning rate
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_stepsize', nargs='+', type=int, default=[100, 200], help='learning rate decay step size')
parser.add_argument('--lr_gamma', default=0.1, type=float, help='gamma for learning rate decay')
###################### Changes ###############################
parser.add_argument('--Dours',
                    action='store_true',
                    default=False,
                    help='read ours data')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# tensorboard Path -----------------------
writer_path = args.tensorboard_path
if args.SF3D:
    writer_path += '_SF3D'
if args.real:
    writer_path += '_real'
############### Changes #######################
if args.Dours:
    writer_path += '_Dours'
##############################################
writer = SummaryWriter(writer_path)
# -----------------------------------------

# import dataloader ------------------------------
# from dataloader import filename_loader as lt
from dataloader import filename_loader_ours as lt
if args.real:
    from dataloader import grayscale_Loader as DA
    print("Real World image loaded!!!")
else:
    from dataloader import RGB_Loader as DA
    print("Synthetic data image loaded!!!")
# -------------------------------------------------

# Random Seed -----------------------------
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# ------------------------------------------
# ------------------ OUR DATA 256 * 1024-------------------------------------------------
angle_y = np.array([(i - 0.5) / 256 * 180 for i in range(128, -128, -1)])
angle_ys = np.tile(angle_y[:, np.newaxis, np.newaxis], (1, 1024, 1  ))
equi_info = angle_ys
Equi_infos = equi_info
# -------------------------------------------------------------------

# Load Data ---------------------------------------------------------
# ________________________
train_up_img, train_down_img, train_up_disp, valid_up_img, valid_down_img, valid_up_disp = lt.dataloader(
    args.datapath, args.datapath2)

Equi_infos = equi_info
TrainImgLoader = torch.utils.data.DataLoader(DA.myImageFolder(
    Equi_infos, train_up_img, train_down_img, train_up_disp, True),
                                             batch_size=args.batch,
                                             shuffle=True,
                                             num_workers=8,
                                             drop_last=False)

ValidImgLoader = torch.utils.data.DataLoader(DA.myImageFolder(
    Equi_infos, valid_up_img, valid_down_img, valid_up_disp, False),
                                             batch_size=args.batch,
                                             shuffle=False,
                                             num_workers=4,
                                             drop_last=False)
# -----------------------------------------------------------------------------------------

# Load model ----------------------------------------------
if args.model == '360SDNet':
    model = LCV_ours_sub3(args.maxdisp, maxdepth=args.maxdepth)
else:
    raise NotImplementedError('Model Not Implemented!!!')
# ----------------------------------------------------------

# assign initial value of filter cost volume ---------------------------------
init_array = np.zeros((1, 1, 7, 1))  # 7 of filter
init_array[:, :, 3, :] = 28. / 540
init_array[:, :, 2, :] = 512. / 540
model.forF.forfilter1.weight = torch.nn.Parameter(torch.Tensor(init_array))
# -----------------------------------------------------------------------------

# Multi_GPU for model ----------------------------
if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()
# -------------------------------------------------

# Load Checkpoint -------------------------------
start_epoch = 0
if args.checkpoint is not None:
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict['state_dict'])
    start_epoch = state_dict['epoch']
    # load pretrain from MP3D for SF3D
    if start_epoch == 50 and args.SF3D:
        start_epoch = 0
        print("MP3D pretrained 50 epoch for SF3D Loaded!!!")
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
# --------------------------------------------------

# Optimizer ----------
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
scheduler = MultiStepLR(optimizer, milestones=args.lr_stepsize, gamma=args.lr_gamma)

# ---------------------


# Freeze Unfreeze Function
# freeze_layer ----------------------
def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


# if use nn.DataParallel(model), model.module.filtercost
# else use model.filtercost
freeze_layer(model.module.forF.forfilter1)


# Unfreeze_layer --------------------
def unfreeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = True
# ------------------------------------


# Train Function -------------------
def train(imgU, imgD, disp, metric_log, epoch):
    model.train()
    imgU = Variable(torch.FloatTensor(imgU.float()))
    imgD = Variable(torch.FloatTensor(imgD.float()))
    disp = Variable(torch.FloatTensor(disp.float()))
    # cuda?
    if args.cuda:
        imgU, imgD, disp_true = imgU.cuda(), imgD.cuda(), disp.cuda()

    # mask value
    # --------------------------- Changes - ----------------------
    # mask = (disp_true < args.maxdisp) & (disp_true > 0)
    mask = (disp_true < args.maxdepth) & (disp_true > 0)
    mask.detach_()

    optimizer.zero_grad()
    # Loss --------------------------------------------
    output1, output2, output3 = model(imgU, imgD)

    output1 = torch.squeeze(output1, 1)
    output2 = torch.squeeze(output2, 1)
    output3 = torch.squeeze(output3, 1)
    # print("333", output3.shape)
    loss = 0.5 * F.smooth_l1_loss(
        output1[mask], disp_true[mask], size_average=True
    ) + 0.7 * F.smooth_l1_loss(
        output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(
            output3[mask], disp_true[mask], size_average=True)

    with torch.no_grad():
        pred = (output3) * args.scale


    metric_log.calculate(disp_true, pred, loss=loss.item())
    # --------------------------------------------------

    loss.backward()
    optimizer.step()

    return loss.data


# Valid Function -----------------------
def val(imgU, imgD, disp_true, metric_log):
    model.eval()
    imgU = Variable(torch.FloatTensor(imgU.float()))
    imgD = Variable(torch.FloatTensor(imgD.float()))
    # cuda?
    if args.cuda:
        imgU, imgD = imgU.cuda(), imgD.cuda()
    # mask value
    # ------------------  ----------------
    mask = (disp_true < args.maxdepth) & (disp_true > 0)
    # ------------------  ----------------

    with torch.no_grad():
        output3 = model(imgU, imgD)

    output = torch.squeeze(output3.data.cpu(), 1)
    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        # loss_epe = torch.mean(torch.abs(output[mask] - disp_true[mask]))  # end-point-error
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    # computing 3-px error#
    metric_log.calculate(disp_true, output, loss=loss.item())

    torch.cuda.empty_cache()


    return loss, output


# Adjust Learning Rate
def adjust_learning_rate(optimizer, epoch):

    lr = 0.001
    if epoch > args.start_decay:
        lr = 0.0001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


best_RMSE = 1e10
# Main Function ----------------------------------
def main():
    global best_RMSE
    global_step = 0
    global_val = 0

    # set logger
    log = logger.setup_logger(os.path.join(args.save_path, 'training.log'))
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    # writer = SummaryWriter(args.save_path + '/tensorboardx')

    # Start Training -----------------------------
    start_full_time = time.time()
    for epoch in tqdm(range(start_epoch + 1, args.epochs + 1), desc='Epoch'):
        # -------------------------
        scheduler.step()
        train_metric = utils_func.Metric()
        # -------------------------
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)

        # unfreeze filter --------------
        if epoch >= args.start_learn:
            unfreeze_layer(model.module.forF.forfilter1)
        # -------------------------------

        # Train ----------------------------------
        for batch_idx, (imgU_crop, imgD_crop,
                        disp_crop) in tqdm(enumerate(TrainImgLoader),
                                           desc='Train iter'):
            loss = train(imgU_crop, imgD_crop, disp_crop, train_metric, epoch)
            total_train_loss += loss
            global_step += 1
            writer.add_scalar('loss', loss, global_step)  # tensorboardX for iter


        writer.add_scalar('total train loss',
                          total_train_loss / len(TrainImgLoader),
                          epoch)  # tensorboardX for epoch

        log.info(train_metric.print(0, 'TRAIN Epoch' + str(epoch)))
        train_metric.tensorboard(writer, epoch, token='TRAIN')

        # ----------------------------------------------------

        # Save Checkpoint ------------------------------------
        if not os.path.isdir(args.save_checkpoint):
            os.makedirs(args.save_checkpoint)
        if args.save_checkpoint[-1] == '/':
            args.save_checkpoint = args.save_checkpoint[:-1]
        savefilename = args.save_checkpoint + '/checkpoint_' + str(
            epoch) + '.tar'
        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
            }, savefilename)
        # --------------------------------------------------------

        # Valid --------------------------------------------------
        test_metric = utils_func.Metric()

        total_val_loss = 0
        total_val_crop_rmse = 0
        for batch_idx, (imgU, imgD, disp) in tqdm(enumerate(ValidImgLoader),
                                                  desc='Valid iter'):
            val_loss, val_output = val(imgU, imgD, disp, test_metric)

            # for depth cropped rmse -------------------------------------
            mask_de_gt = disp > 0
            val_crop_rmse = np.sqrt(
                np.mean(((val_output.data.cpu().numpy())[mask_de_gt] - np.array(disp[mask_de_gt]))**2))
            # -------------------------------------------------------------
            # Loss ---------------------------------
            total_val_loss += val_loss
            total_val_crop_rmse += val_crop_rmse
            # ---------------------------------------
            # Step ------
            global_val += 1
            # ------------
        writer.add_scalar('total validation loss_org',
                          total_val_loss / (len(ValidImgLoader)),
                          epoch)  # tensorboardX for validation in epoch
        writer.add_scalar('total validation crop 26 depth rmse_org',
                          total_val_crop_rmse / (len(ValidImgLoader)),
                          epoch)  # tensorboardX rmse for validation in epoch


        log.info(test_metric.print(0, 'TEST Epoch' + str(epoch)))
        test_metric.tensorboard(writer, epoch, token='TEST')

    # SAVE
    is_best = test_metric.RMSELIs.avg < best_RMSE
    best_RMSE = min(test_metric.RMSELIs.avg, best_RMSE)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.model,
        'state_dict': model.state_dict(),
        'best_RMSE': best_RMSE,
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best, epoch, folder=args.save_path)
    writer.close()

    # End Training
    print("Training Ended!!!")
    print('full training time = %.2f HR' %
          ((time.time() - start_full_time) / 3600))
# ----------------------------------------------------------------------------
def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar', folder='result/default'):
    torch.save(state, folder + '/' + filename)
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')
    if args.checkpoint_interval > 0 and (epoch + 1) % args.checkpoint_interval == 0:
        shutil.copyfile(folder + '/' + filename, folder + '/checkpoint_{}.pth.tar'.format(epoch + 1))

if __name__ == '__main__':
    main()
