import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import copy
import time
import numpy as np
import pickle
import argparse
import os.path as osp
from collections import OrderedDict
from misc.utils import mkdir
from models import *
from misc.utils import init_model, init_random_seed, mkdirs, lab_conv
from torch.nn import DataParallel
from models import DenoiseResnet, resnet
from advertorch.context import ctx_noparamgrad_and_eval
import torchvision.utils as vutils

from misc.saver import Saver
from datasets.dataloader import get_data_loader

from pdb import set_trace as st




def train_Ours(args, train_loader, val_loader, knownclass, 
               DeEncoder, DeDecoder, DeNorClsfier, DeSSDClsfier,
               PeerNet,  
               summary_writer, saver):

    seed = init_random_seed(args.manual_seed)

    criterionCls = nn.CrossEntropyLoss()
    criterionFeat = nn.L1Loss()

    if args.recway is 'MSE':
        criterionRec = nn.MSELoss()
    elif args.recway is 'L1':
        criterionRec = nn.L1Loss()
    
    optimizer_de = optim.Adam(list(DeEncoder.parameters())
                            +list(DeNorClsfier.parameters())
                            +list(DeSSDClsfier.parameters())
                            +list(DeDecoder.parameters()), lr=args.lr) 
    optimizer_cl = optim.Adam(PeerNet.parameters(), lr=args.lr)  

    if args.adv is 'PGDattack':
        print("**********Defense PGD Attack**********")
    elif args.adv is 'FGSMattack':
        print("**********Defense FGSM Attack**********")

    if args.adv is 'PGDattack':
        from advertorch.attacks import PGDAttack
        nor_adversary = PGDAttack(predict1=DeEncoder, predict2=DeNorClsfier, nb_iter=args.adv_iter)
        rot_adversary = PGDAttack(predict1=DeEncoder, predict2=DeSSDClsfier, nb_iter=args.adv_iter)

    elif args.adv is 'FGSMattack':
        from advertorch.attacks import GradientSignAttack
        nor_adversary = GradientSignAttack(predict1=DeEncoder, predict2=DeNorClsfier)
        rot_adversary = GradientSignAttack(predict1=DeEncoder, predict2=DeSSDClsfier)

    elif args.adv is 'PATCHattack':
        from advertorch import ROA
        DeEncoder.eval()
        DeNorClsfier.eval()
        roa = ROA.ROA(DeEncoder, DeNorClsfier, args.img_size) 

    global_step = 0
    # ----------
    #  Training
    # ----------
    for epoch in range(args.n_epoch):
        
        DeEncoder.train()
        DeDecoder.train()        
        DeNorClsfier.train()        
        DeSSDClsfier.train() 

        PeerNet.train()         
   
        if args.lr_dacay:
            lr = get_lr(epoch)
            print('lr is: {:.6f}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


        for steps, (oriimg, orilab, rotimg, rotlab) in enumerate(train_loader):

            orilab = lab_conv(knownclass, orilab)
            oriimg, orilab = oriimg.cuda(), orilab.long().cuda()

            rotimg, rotlab = rotimg.cuda(), rotlab.long().cuda()

            if args.adv in ['PGDattack', 'FGSMattack']:
                with ctx_noparamgrad_and_eval(DeEncoder):
                    with ctx_noparamgrad_and_eval(DeNorClsfier):
                        with ctx_noparamgrad_and_eval(DeSSDClsfier):
                            oriimg_adv = nor_adversary.perturb(oriimg, orilab)
                            rotimg_adv = rot_adversary.perturb(rotimg, rotlab)
            elif args.adv is 'PATCHattack':
                DeEncoder.eval()
                DeNorClsfier.eval() 
                oriimg_adv = roa.gradient_based_search(oriimg, orilab, alpha=0.05,\
                num_iter=args.adv_iter, width=5 , height=5, xskip=1, yskip=1, potential_nums=20)
                rotimg_adv = roa.gradient_based_search(rotimg, rotlab, alpha=0.05,\
                num_iter=args.adv_iter, width=5 , height=5, xskip=1, yskip=1, potential_nums=20)
                DeEncoder.train()
                DeNorClsfier.train() 

            #*********** adv **************

            # ori classification
            latent_feat_de = DeEncoder(oriimg_adv)
            norpred_de =  DeNorClsfier(latent_feat_de)
            norlossCls = criterionCls(norpred_de, orilab)

            # reconstruction
            recon = DeDecoder(latent_feat_de)
            lossRec = criterionRec(recon, oriimg)

            # rot classification
            ssdpred = DeSSDClsfier(DeEncoder(rotimg_adv))
            rotlossCls = criterionCls(ssdpred, rotlab)

            norpred_cl = PeerNet(oriimg)

            lossKLD = F.kl_div(F.log_softmax(norpred_de, dim=1),F.softmax(norpred_cl.detach(), dim=1),False)/norpred_de.shape[0]

            loss_de = args.norClsWgt*norlossCls + args.rotClsWgt*rotlossCls + \
                    args.RecWgt*lossRec + args.KLDWgt*lossKLD

            optimizer_de.zero_grad()
            loss_de.backward()
            optimizer_de.step()

            #*********** clean **************
            norpred_de = DeNorClsfier(DeEncoder(oriimg_adv))

            # ori classification
            norlossCls = criterionCls(norpred_cl, orilab)
            lossKLD = F.kl_div(F.log_softmax(norpred_cl, dim=1),F.softmax(norpred_de.detach(), dim=1),False)/norpred_cl.shape[0]

            loss_cl = args.norClsWgt*norlossCls + args.KLDWgt*lossKLD

            optimizer_cl.zero_grad()
            loss_cl.backward()
            optimizer_cl.step()



            #============ tensorboard the log info ============#
            lossinfo = {
                'loss_de': loss_de.item(),               
                'norlossCls': norlossCls.item(), 
                'lossRec': lossRec.item(),                                                                                     
                'rotlossCls': rotlossCls.item(),                                                                                     
                'lossKLD': lossKLD.item(),                                                                                     
                'loss_cl': loss_cl.item(),                                                                                     
                    } 
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, global_step) 

            global_step+=1
   
            #============ print the log info ============# 
            if (steps) % args.log_step == 0:
                errors = OrderedDict([('loss_de', loss_de.item()),
                                    ('norlossCls', norlossCls.item()),
                                    ('lossRec', lossRec.item()),
                                    ('rotlossCls', rotlossCls.item()),
                                    ('lossKLD', lossKLD.item()),
                                    ('loss_cl', loss_cl.item()),
                                        ])        
                saver.print_current_errors((epoch), (steps), errors) 


        # evaluate performance on validation set periodically
        if ((epoch) % args.val_epoch == 0):

            # switch model to evaluation mode
            DeEncoder.eval()
            DeNorClsfier.eval()

            running_corrects = 0.0
            epoch_size = 0.0
            val_loss_list = []

            # calculate accuracy on validation set
            for steps, (oriimg, orilab) in enumerate(val_loader):

                orilab = lab_conv(knownclass, orilab)
                oriimg, orilab = oriimg.cuda(), orilab.long().cuda()

                if args.adv in ['PGDattack', 'FGSMattack']:
                    oriimg_adv = nor_adversary.perturb(oriimg, orilab)
                elif args.adv is 'PATCHattack':
                    oriimg_adv = roa.gradient_based_search(oriimg, orilab, alpha=0.05,\
                    num_iter=args.adv_iter, width=5 , height=5, xskip=1, yskip=1, potential_nums=20)

                with torch.no_grad():
                    logits = DeNorClsfier(DeEncoder(oriimg_adv))
                    _, preds = torch.max(logits, 1)
                    running_corrects += torch.sum(preds == orilab.data)
                    epoch_size += oriimg.size(0)
                    
                    val_loss = criterionCls(logits, orilab)

                    val_loss_list.append(val_loss.item())

            val_loss_mean = sum(val_loss_list)/len(val_loss_list)

            val_acc =  running_corrects.double() / epoch_size
            print('Val Acc: {:.4f}, Val Loss: {:.4f}'.format(val_acc, val_loss_mean))

            valinfo = {
                'Val Acc': val_acc.item(),               
                'Val Loss': val_loss.item(), 
                    } 
            for tag, value in valinfo.items():
                summary_writer.add_scalar(tag, value, (epoch))


        
        if ((epoch) % args.model_save_epoch == 0):
            model_save_path = os.path.join(args.results_path, args.training_type, 
                            'snapshots', args.datasetname+'-'+args.split, args.denoisemean, 
                             args.adv+str(args.adv_iter))    
            mkdir(model_save_path) 
            torch.save(DeEncoder.state_dict(), os.path.join(model_save_path,
                "DeEncoder-{}.pt".format(epoch)))
            torch.save(DeNorClsfier.state_dict(), os.path.join(model_save_path,
                "DeNorClsfier-{}.pt".format(epoch)))
            torch.save(DeDecoder.state_dict(), os.path.join(model_save_path,
                "DeDecoder-{}.pt".format(epoch)))



    torch.save(DeEncoder.state_dict(), os.path.join(model_save_path, "DeEncoder-final.pt"))
    torch.save(DeNorClsfier.state_dict(), os.path.join(model_save_path, "DeNorClsfier-final.pt"))
    torch.save(DeDecoder.state_dict(), os.path.join(model_save_path, "DeDecoder-final.pt"))


def get_lr(epoch):
    if epoch <= args.n_epoch * 0.3:
        return args.lr
    elif epoch <= args.n_epoch * 0.6:
        return args.lr * 0.1
    else:
        return args.lr * 0.01


        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="advOpenset")
    
    parser.add_argument('--adv', type=str, default='FGSMattack') #clean PGDattack FGSMattack PATCHattack
    parser.add_argument('--adv_iter', type=int, default=5)

    parser.add_argument('--results_path', type=str, default='./results/')
    parser.add_argument('--training_type', type=str, default='Ours_FD3')

    parser.add_argument('--parallel_train', type=str, default=True) # cifar10 svhn False; tinyimagenet True 
    parser.add_argument('--datasetname', type=str, default='imagenet') # cifar10 tinyimagenet svhn imagenet
    parser.add_argument('--split', type=str, default='0')
    parser.add_argument('--img_size', type=int, default=64)  # cifar svhn 32 tinyimagenet 64 imagenet 128
    parser.add_argument('--input_chlnum', type=int, default=3) 

    parser.add_argument('--denoisemean', type=str, default='cbam') 
    parser.add_argument('--init_type', type=str, default='normal') # normal xavier kaiming


    parser.add_argument('--denoise', type=str, default=[True, True, True, True, True]) 
    # de_adv

    parser.add_argument('--recway', type=str, default='MSE') 
    parser.add_argument('--norClsWgt', type=int, default=1)
    parser.add_argument('--rotClsWgt', type=int, default=1) 
    parser.add_argument('--RecWgt', type=int, default=1)
    parser.add_argument('--KLDWgt', type=int, default=2)
    parser.add_argument('--latent_size', type=int, default=512)

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_dacay', type=str, default=False) 
    parser.add_argument('--n_epoch', type=int, default=120) # cifar svhn 120 tinyimagenet 150
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--model_save_epoch', type=int, default=2)
    parser.add_argument('--val_epoch', type=int, default=2)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--manual_seed', type=int, default=None)

    args = parser.parse_args()
    print(args)

    log_file = os.path.join(args.results_path, args.training_type, 'log', 
        args.datasetname+'-'+args.split, args.denoisemean, args.adv+str(args.adv_iter))

    
    summary_writer = SummaryWriter(log_file)
    saver = Saver(args, log_file)
    saver.print_config()

    train_loader, val_loader, knownclass = get_data_loader(name=args.datasetname, train=True, split=args.split, 
                                    batch_size=args.batchsize, image_size=args.img_size)

    nclass = len(knownclass)

    DeEncoder = init_model(net=DenoiseResnet.ResnetEncoder(input_chlnum=args.input_chlnum, denoisemean=args.denoisemean, 
                    latent_size= args.latent_size, denoise=args.denoise), 
                    init_type = args.init_type, restore=None)

    DeDecoder = init_model(net=DenoiseResnet.ResnetDecoder(latent_size= args.latent_size), 
                    init_type = args.init_type, restore=None)

    DeNorClsfier = init_model(net=DenoiseResnet.NorClassifier(latent_size= args.latent_size, num_classes=nclass), 
                    init_type = args.init_type, restore=None)

    DeSSDClsfier = init_model(net=DenoiseResnet.SSDClassifier(latent_size= args.latent_size), 
                    init_type = args.init_type, restore=None)


    PeerNet = init_model(net=resnet.ResNet18(num_classes=nclass), 
                    init_type = args.init_type, restore=None)

    train_Ours(args, train_loader, val_loader, knownclass, 
               DeEncoder, DeDecoder, DeNorClsfier, DeSSDClsfier,
               PeerNet, 
               summary_writer, saver)


