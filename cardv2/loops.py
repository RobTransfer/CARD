from __future__ import print_function, division
from algorithm.tools import *
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .util import AverageMeter
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        if not p_s.size(1) == p_t.size(1):
            p_s = F.interpolate(p_s.unsqueeze(1),size=p_t.size(1),mode='nearest')
            p_s = p_s.squeeze(2)
        print(p_s.size(),p_t.size())
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()
def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt,attack,s):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    for idx, data in enumerate(train_loader):
        
        input, target, index, contrast_idx = data
        #try List
        data_time.update(time.time() - end)

        input = input.float()
        input = input.unsqueeze(1)
        input = input.unsqueeze(3)
        if torch.cuda.is_available():
            input = input.cuda()
            adv = attack.generate(input.cpu().numpy(),target.cpu().numpy())
            adv = torch.from_numpy(adv).cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.Mode == 'contrastive':
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        if s != opt.input_dim:
            input_t = criterion_kd.embed_in_s(input).view(input.shape[0],input.shape[1],opt.input_dim,input.shape[3])
            logit_t = model_t(input_t)
        else:
            logit_t = model_t(input)
        logit_s = model_s(adv)
        logit_sog = model_s(input)
        if s != opt.input_dim:
            adv_t = criterion_kd.embed_in_s(adv).view(adv.shape[0],adv.shape[1],opt.input_dim,adv.shape[3])
        with torch.no_grad():
            if s != opt.input_dim:
                _,feat_t = model_t(adv_t,opt.is_sample)
                feat_t=feat_t.detach()
            else:
                _,feat_t = model_t(adv,opt.is_sample)
                feat_t=feat_t.detach()
        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        ogloss = criterion_cls(logit_sog, target)
        if opt.input_dim == s and logit_s.shape[1] == logit_t.shape[1]:
            loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        
        
        _,f_s = model_s(adv,opt.is_sample)
        f_t = feat_t
        loss_kd = criterion_kd(f_s, f_t, index, contrast_idx) 
        if logit_s.shape[1] == logit_t.shape[1]:
            loss = opt.gamma * loss_cls + opt.beta * loss_kd + opt.alpha* ogloss + opt.cof * loss_div
        else:
            loss = opt.gamma * loss_cls + opt.beta * loss_kd + opt.alpha* ogloss
        pred = torch.max(logit_s,dim =1)[1]
        acc1 = evaluate(pred.cpu().numpy(),target.cpu().numpy())
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_s.parameters(),1)
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.n_eval_step == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} '
          .format(top1=top1))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, num,opt,attack):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    adv1 = AverageMeter()
    
    totalf1 = 0.0
    totalrecall = 0.0
    totalprecision = 0.0
    fnr = 0.0
    fpr= 0.0
    batchnum = 0
    totalf1adv = 0.0
    totalrecalladv = 0.0
    totalprecisionadv = 0.0
    fnradv = 0.0
    fpradv= 0.0
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                input = input.unsqueeze(1)
                input = input.unsqueeze(3)
                target = target.cuda()

            # compute output
            output = model(input)
            with torch.enable_grad():
                adv = attack.generate(input.cpu().numpy(),target.cpu().numpy())
                adv = torch.from_numpy(adv).cuda()
            model.eval()
            advout = model(adv)
            loss = criterion(output, target)
            output = torch.max(output,dim=1)[1]
            advout = torch.max(advout,dim=1)[1]
            # measure accuracy and record loss
            acc1= evaluate(output.cpu().numpy(), target.cpu().numpy())
            aa = evaluate(advout.cpu().numpy(), target.cpu().numpy())
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))
            if opt.is_binary:
                f1 = f1_score(target.cpu().numpy(),output.cpu().numpy())
                totalf1 += f1
                
                p=precision_score(target.cpu().numpy(),output.cpu().numpy())
                totalprecision+=p
                
                r = recall_score(target.cpu().numpy(),output.cpu().numpy())
                totalrecall +=r
                fp,fn = fpandfn(output.cpu().numpy(), target.cpu().numpy())
                fnr+= fp
                fpr+= fn
                f2=f1_score(target.cpu().numpy(),advout.cpu().numpy())
                totalf1adv += f2
                p2 = precision_score(target.cpu().numpy(),advout.cpu().numpy())
                totalprecisionadv+=p2
                r2=recall_score(target.cpu().numpy(),advout.cpu().numpy())
                totalrecalladv +=r2
                
                fpa,fna = fpandfn(advout.cpu().numpy(), target.cpu().numpy())
                fnradv+= fpa
                fpradv+= fna
            else:
                output[torch.where(output != 0)] = 1
                advout[torch.where(advout != 0)] = 1
                f1 = f1_score(target.cpu().numpy(),output.cpu().numpy())
                totalf1 += f1
                
                p=precision_score(target.cpu().numpy(),output.cpu().numpy())
                totalprecision+=p
                
                r = recall_score(target.cpu().numpy(),output.cpu().numpy())
                totalrecall +=r
                fp,fn = fpandfn(output.cpu().numpy(), target.cpu().numpy())
                fnr+= fp
                fpr+= fn
                f2=f1_score(target.cpu().numpy(),advout.cpu().numpy())
                totalf1adv += f2
                p2 = precision_score(target.cpu().numpy(),advout.cpu().numpy())
                totalprecisionadv+=p2
                r2=recall_score(target.cpu().numpy(),advout.cpu().numpy())
                totalrecalladv +=r2
                
                fpa,fna = fpandfn(advout.cpu().numpy(), target.cpu().numpy())
                fnradv+= fpa
                fpradv+= fna
                
            adv1.update(aa,input.size(0))
            batchnum+=1
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not opt.is_binary:
                if idx % opt.n_eval_step == 0:
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Robust {adv1.val:.3f} ({adv1.avg:.3f})'.format(
                        idx, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1,adv1 = adv1))
            else:
                if idx % opt.n_eval_step == 0:
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Robust {adv1.val:.3f} ({adv1.avg:.3f})\t'
                        'F1 {totalf1}\t'
                        'F1adv {totalf12}\t'
                        'precision {totalprecision}\t'
                        'precisionadv {totalprecision2}\t'
                        'recall{totalrecall}\t'
                        'recalladv{totalrecall2}\t'
                        'fpr {totalfpr}\t'
                        'fpradv {totalfpr2}\t'
                        'fnr {totalfnr}\t'
                        'fnradv {totalfnr2}\t'.format(
                        idx, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1,adv1 = adv1,totalf1=f1,totalf12=f2,totalprecision=p,totalprecision2=p2,totalrecall=r,totalrecall2=r2,totalfpr=fp,totalfpr2=fpa,totalfnr=fn,totalfnr2=fna))
        if not opt.is_binary:
            print(' * Acc@1 {top1.avg:.3f} Robust {adv1.avg:.3f}'
                .format(top1=top1,adv1 = adv1))
        else:
            all = [totalf1/batchnum,totalf1adv/batchnum,totalprecision/batchnum,totalprecisionadv/batchnum,totalrecall/batchnum,totalrecalladv/batchnum,fpr/batchnum,fpradv/batchnum,fnr/batchnum,fnradv/batchnum]
            print(' * Acc@1 {top1.avg:.3f} Robust {adv1.avg:.3f} F1:{all[0]}, F1adv:{all[1]}, Precision:{all[2]}, Precisionadv:{all[3]},Recall:{all[4]}, recalladv:{all[5]},FPR:{all[6]}, FPRadv:{all[7]},FNR: {all[8]}, FNRadv{all[9]}'
                .format(top1=top1,adv1 = adv1,all = all))

    return top1.avg, adv1.avg,losses.avg
