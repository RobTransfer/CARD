import numpy as np
import torch
import torchvision 
import torchvision.transforms as transforms
import torch.nn as nn
import Distillaton.AAD.utils as utils
from FineTuning.Twins.robustness.tools.custom_modules import SequentialWithArgs
import torch.backends.cudnn as cudnn
import pandas
from datetime import datetime
from torchvision.models import wide_resnet50_2
from FineTuning.Twins.transfer_utils import fine_tunify, transfer_datasets
from FineTuning.Twins.robustness import datasets, model_utils
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from cardv2.helper.util import AverageMeter
from model.densenet import DenseNet
from model.mobilenet import MobileNet
from model.wideresnet import WideResNet 
from model.resnet18 import ResNet18
import pickle
import time
from tqdm import tqdm
import os
from cardv2.dataset.pairloader_maggie import *
from art.attacks.evasion import ProjectedGradientDescent
from others.cutout import Cutout
from Distillaton.AAD.AT_helper import Madry_PGD, adaad_inner_loss
from advertorch.attacks import LinfPGDAttack
from algorithm.args import *
from algorithm.tools import AttackPGD,evaluate,generate_atk,accuracy,fpandfn1,precision_score1, recall_score1, f1_score1
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
args = arg_gen()
if args.cardtrain == 'old':
    from cardv2.helper.loops import validate,train_distill,DistillKL
    from cardv2.crd.criterion import CRDLoss
else:
    from cardv2.helper.loopsv2 import validate,train_distill,DistillKL
    from cardv2.crd.criterionv2 import CRDLoss
device = f"cuda:{args.device}" if torch.cuda.is_available else "cpu"
torch.cuda.set_device(args.device)

if not os.path.exists(args.model_folder):
    os.makedirs(args.model_folder)
if not os.path.exists(args.featurepath):
    os.makedirs(args.featurepath)
minv = 0
maxv = 0
npydir = None
transform = None
dataset = None
if args.dataset == "cifar10":
    npydir = f"robust_{args.dataset}_feats.npy"
    transform = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                Cutout(n_holes=1, length=16)
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
        }
    numofclass = 10
    train_dataset = torchvision.datasets.CIFAR10(root='/home/huan1932/data/CIFAR-10', train=True, transform=transform["train"], download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='/home/huan1932/data/CIFAR-10', train=False, transform=transform["val"], download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
elif args.dataset == "cifar100":
    npydir = f"robust_{args.dataset}_feats.npy"
    transform = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                Cutout(n_holes=1, length=16)
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
        }
    numofclass = 100
    train_dataset = torchvision.datasets.CIFAR100(root='/home/huan1932/data/CIFAR-100', train=True, transform=transform["train"], download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='/home/huan1932/data/CIFAR-100', train=False, transform=transform["val"], download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
else:
    if args.dataset in ['data/UNSW-NB15/trainNE.csv','data/UNSW-NB15/traineandb.csv','data/UNSW-NB15/train.csv']:
        dataset = 'UNSW'
    elif args.dataset in ['data/NSL-KDD/kddbinarytrain.csv','data/NSL-KDD/kddmultitrain.csv']:
        dataset = 'KDD'
    npydir = f"robust_{dataset}_feats.npy"
    train_loader, test_loader,train_dataset,numofclass ,n_data = get_dataloaders_sample(args,args.dataset,args.testdata,batch_size=args.batch, num_workers=8, k=args.nce_k, mode='exact',is_sample=args.is_sample, percent=args.percent)
    # keys = args.dataset.split("/")
    # if "UNSW-NB15" in keys:
    #     # mapping = {6:0,0:1,1:2,2:3,4:4,5:5,7:6,8:7,9:8}
    #     ##['Analysis'1 'Backdoor'2 'DoS'3 'Fuzzers'4 'Generic'5 'Normal0''Reconnaissance'6 'Shellcode'7 'Worms'8]
    #     data = pandas.read_csv(args.dataset)
    #     datatest = pandas.read_csv(args.testdata)
    #     data = data.drop_duplicates()
    #     datatest = datatest.drop_duplicates()
    #     data1= data.drop(data[data['attack_cat'] == 0].index)
        
    #     data = data[data['attack_cat'] == 0].sample(n=35000,replace=False)
        
    #     data=pandas.concat([data1,data],ignore_index=True,axis=0)
        
    #     if args.how == 'm':
    #         x_train = data.iloc[:,:-2].values
    #         # data['attack_cat'] = data['attack_cat'].map(mapping)
    #         y_train = data.iloc[:,-1].values
    #         x_test =datatest.iloc[:,:-2].values
    #         # datatest['attack_cat'] = datatest['attack_cat'].map(mapping)
    #         y_test = datatest.iloc[:,-1].values
    #     elif args.how == 'b':
    #         x_train = data.iloc[:,:-2].values
    #         y_train = data.iloc[:,-2].values
    #         x_test =datatest.iloc[:,:-2].values
    #         y_test = datatest.iloc[:,-2].values
    #     numofclass = len(np.unique(y_train))
    #     n_data = x_train.shape[0]
    #     train = [torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long)]
    #     test = [torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)]
    #     train_dataset = TensorDataset(train[0],train[1])
    #     test_dataset = TensorDataset(test[0],test[1])
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    # if args.targettrain != None and args.targettest != None:
        
        # # mappingtar = {3:1,6:0}
        # datat = pandas.read_csv(args.targettrain)
        # datatestt = pandas.read_csv(args.targettest)
        # x_traint = datat.iloc[:,:-1].values
        # # datat['attack_cat'] = datat['attack_cat'].map(mappingtar)
        # y_traint = datat.iloc[:,-1].values
        # x_testt =datatestt.iloc[:,:-1].values
        # # datatestt['attack_cat'] = datatestt['attack_cat'].map(mappingtar)
        # y_testt = datatestt.iloc[:,-1].values
        # numofclasstar = len(np.unique(y_traint))
        # train = [torch.tensor(x_traint, dtype=torch.float), torch.tensor(y_traint, dtype=torch.long)]
        # test = [torch.tensor(x_testt, dtype=torch.float), torch.tensor(y_testt, dtype=torch.long)]
        # train_datasett = TensorDataset(train[0],train[1])
        # test_datasett = TensorDataset(test[0],test[1])
        # train_loadert = DataLoader(train_datasett, batch_size=args.batch, shuffle=True)
        # test_loadert = DataLoader(test_datasett, batch_size=args.batch, shuffle=False)
    # else:
    #     data = pandas.read_csv(args.dataset).drop_duplicates()
    #     datatest = pandas.read_csv(args.testdata).drop_duplicates()
    
    #     x_train = data.iloc[:,:-1].values
    #     y_train = data.iloc[:,-1].values
    #     x_test =datatest.iloc[:,:-1].values
    #     y_test = datatest.iloc[:,-1].values
        
    #     numofclass = len(np.unique(y_train))
    #     n_data = x_train.shape[0]
    #     train = [torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long)]
    #     test = [torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)]
    #     train_dataset = TensorDataset(train[0],train[1])
    #     test_dataset = TensorDataset(test[0],test[1])
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    #     if args.targettrain != None and args.targettest != None:
    #         datat = pandas.read_csv(args.targettrain)
    #         datatestt = pandas.read_csv(args.targettest)
    #         x_traint = datat.iloc[:,:-1].values
    #         y_traint = datat.iloc[:,-1].values
    #         x_testt =datatestt.iloc[:,:-1].values
    #         y_testt = datatestt.iloc[:,-1].values
    #         numofclasstar = len(np.unique(y_traint))
    #         n_data = x_traint.shape[0]
    #         train = [torch.tensor(x_traint, dtype=torch.float), torch.tensor(y_traint, dtype=torch.long)]
    #         test = [torch.tensor(x_testt, dtype=torch.float), torch.tensor(y_testt, dtype=torch.long)]
    #         train_dataset = TensorDataset(train[0],train[1])
    #         test_dataset = TensorDataset(test[0],test[1])
    #         train_loadert = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    #         test_loadert = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
data_time = AverageMeter()
def lr_decay(epoch, total_epoch):
        if args.lr_schedule == 'piecewise':
            if total_epoch == 200:
                epoch_point = [100, 150]
            elif total_epoch == 110: 
                epoch_point = [100, 105] # Early stop for Madry adversarial training
            elif total_epoch ==50:
                epoch_point = [25, 40]
            else:
                epoch_point = [int(total_epoch/2),int(total_epoch-5)]
            if epoch < epoch_point[0]:
                if args.warmup_lr and epoch < args.warmup_lr_epoch:
                    return 0.001 + epoch / args.warmup_lr_epoch * (args.lr-0.001)
                return args.lr
            if epoch < epoch_point[1]:
                return args.lr / 10
            else:
                return args.lr / 100
        elif args.lr_schedule == 'cosine':
            if args.warmup_lr:
                if epoch < args.warmup_lr_epoch:
                    return 0.001 + epoch / args.warmup_lr_epoch * (args.lr-0.001)
                else:
                    return np.max([args.lr * 0.5 * (1 + np.cos((epoch-args.warmup_lr_epoch) / (total_epoch-args.warmup_lr_epoch) * np.pi)), 1e-4])
            return np.max([args.lr * 0.5 * (1 + np.cos(epoch / total_epoch * np.pi)), 1e-4])
        elif args.lr_schedule == 'constant':
            return args.lr
        else:
            raise NotImplementedError  
if not args.is_sample: 
    for data, label in train_loader:
        x = torch.max(data).item()
        y = torch.min(data).item()
        maxv = x if max == 0 else max(maxv, x)
        minv = y if min == 0 else min(minv, y)
else:
    for data, label,index,contraindex in train_loader:
        x = torch.max(data).item()
        y = torch.min(data).item()
        maxv = x if max == 0 else max(maxv, x)
        minv = y if min == 0 else min(minv, y)
# if args.Mode == 'contrastive':
#     if args.learning_model == 'res':
#         model = ResNet18(args = args, numofclass=numofclasstar).to(device)
#     elif args.learning_model == 'wide':
#         model = WideResNet(args,num_classes=numofclasstar,depth=args.depth,widen_factor=args.widening).to(device)
#         # model = wide_resnet50_2(num_classes=numofclasstar).to(device)
#     else:
#         model = torchvision.models.resnet50().to(device)
# else:
if args.learning_model == 'res':
    model = ResNet18(args = args, numofclass=numofclass).to(device)
elif args.learning_model == 'wide':
    model = WideResNet(args,num_classes=numofclass,depth=args.depth,widen_factor=args.widening).to(device)
    # model = wide_resnet50_2(num_classes=numofclass).to(device)
elif args.learning_model == 'mobilenet':
    model = MobileNet(args=args,num_classes=numofclass).to(device)
else:
    model = torchvision.models.resnet50().to(device)
    
##------------other model ---------------------------

end = time.time()
if args.Mode == "standard":
    
    def test(model,loader,attack = None):
        if attack == None:
            model.eval()
            total_correct=0
            total_loss=0
            total = 0
            with torch.no_grad():
                for images,labels in loader:
                    if args.datatype == 'network':
                        images = images.unsqueeze(1)
                        images = images.unsqueeze(3)
                    images=images.to(device)
                    labels=labels.to(device)
                    preds=model(images)
                    loss=F.cross_entropy(preds,labels)
                    total_loss+=loss.item()
                    preds = torch.max(preds,dim =1)[1]
                    total_correct+=evaluate(preds.cpu().numpy(), labels.cpu().numpy(), 'sum')
                    total += labels.size(0)
            model.train()
            return total_loss, abs((total_correct/total))
        else:
            total_acc = 0.0
            total_n = 0
            num = 0
            total_adv_acc = 0.0
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            total_fpr = 0
            total_fnr = 0
            total_loss=0
            model.eval()
            with torch.no_grad():
                for data, label in loader:
                    data, label = data.to(device), label.to(device)
                    if args.datatype == 'network':
                        data = data.unsqueeze(1)
                        data = data.unsqueeze(3)
                    output = model(data)
                    loss=F.cross_entropy(output,label)
                    total_loss+=loss.item()
                    pred = torch.max(output, dim=1)[1]
                    
                    te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    
                    total_acc += te_acc
                    num += output.shape[0]
                    noise = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=data.shape), dtype=torch.float32).to(data.device)
                    outputn = model(data+noise)
                    predn = torch.max(outputn, dim=1)[1]
                    n_acc = evaluate(predn.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_n += n_acc
                    
                    pred = (pred != 0)
                    tp = ((pred == 1) & (label == 1)).sum().item()
                    fp = ((pred == 1) & (label == 0)).sum().item()
                    tn = ((pred== 0) & (label == 0)).sum().item()
                    fn = ((pred == 0) & (label == 1)).sum().item()
                    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
                    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1
                    total_fpr += fpr
                    total_fnr += fnr
                    # use predicted label as target label
                    adv_data = torch.rand_like(data)
                    with torch.enable_grad():
                        if args.target != False:
                            a = attack.perturb(data[label == 0], y=torch.randint(1,numofclass).item())
                            b = attack.perturb(data[label != 0], y=0)
                            adv_data[label == 0] = a
                            adv_data[label != 0] = b
                        else:
                            adv_data = attack.perturb(data)
                        # if args.target == True:
                        #     adv_data[label!=0] = attack.generate(data[label!=0].cpu().numpy(), y=np.zeros((data[label!=0].shape[0],)))
                        #     adv_data[label==0] = attack.generate(data[label==0].cpu().numpy(), y=torch.randint(1,numofclass,size = (data[label==0].shape[0],)))
                        #     adv_data = torch.from_numpy(adv_data).float().to(device)
                        # else:
                        #     adv_data = attack.generate(data.cpu().numpy())
                        #     adv_data = torch.from_numpy(adv_data).float().to(device)
                    model.eval()
                    adv_output = model(adv_data)
                    #print(adv_output.size())
                    #print(adv_output[0,:])
                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
            avg_precision = total_precision / len(loader)
            avg_recall = total_recall / len(loader)
            avg_f1 = total_f1 / len(loader)
            avg_fpr = total_fpr / len(loader)
            avg_fnr = total_fnr / len(loader)
            model.train()
            print("StdAcc: ",abs(total_acc / num ) , "AdvAcc: ",abs(total_adv_acc / num),"NaturacorrAcc: ", abs(total_n/num) ,'Precision: ',avg_precision,'Recall: ',avg_recall,'F1: ',avg_f1,'FNR: ',avg_fnr,'FPR: ',avg_fpr)
            return abs((total_acc / num )) , abs((total_adv_acc / num)),total_loss
            
        


    def train(attack):
        if not os.path.exists("./state_dicts"):
            os.mkdir("state_dicts")
        model.train()
        print("Initializing Network")
        if args.dataset in ['data/UNSW-NB15/train.csv','data/UNSW-NB15/traineandb.csv','data/UNSW-NB15/train.csv','data/UNSW-NB15/trainNE.csv','/home/huan1932/NEWCARD/data/UNSW-NB15/trainNE.csv']:
            dataset = 'UNSW'
        elif args.dataset in ['cifar10','cifar100']:
            dataset = args.dataset
        elif args.dataset in ['data/NSL-KDD/kddbinarytrain.csv','data/NSL-KDD/kddmultitrain.csv']:
            dataset = 'KDD'
        trainloader=train_loader
        testloader=test_loader
        #3.prepare optimizer
        # optimizer=optim.Adam(model.parameters(),lr=args.lr)
        optimizer = optim.SGD(model.parameters(),lr= args.lr)
        # optimizer.load_state_dict(info['optimizer'])
        # scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,250,300],gamma=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
        #4.initilazing tensorboard
        comment=f'Wideresnet stdtrain_batch_size={args.batch}_task={args.task_name}_lr={args.lr} device={device}'
        tb=SummaryWriter(comment=comment)
        best_acc=0.0
        for epoch in range(args.epoch):
            train_loss=0
            train_correct=0
            for images,labels in trainloader:
                images=images.to(device)
                data_time.update(time.time() - end)
                #--------maggie add--------
                
                images = images.unsqueeze(1)
                
                images = images.unsqueeze(3)
              
                """ 
                images.shape: torch.Size([1024, 53])
                images.shape: torch.Size([1024, 1, 53])
                images.shape: torch.Size([1024, 1, 53, 1])
                """
                #---------------------------
                
                labels=labels.to(device)
                optimizer.zero_grad()
                
                
                preds=model(images)
            
                
                loss=F.cross_entropy(preds,labels)
                loss.backward()
                optimizer.step()
                train_loss+=loss.item()
                train_correct+=preds.argmax(dim=1).eq(labels).sum().item()
                
            
            present_trainset_acc=train_correct/len(train_dataset)
            present_testset_acc,adv_acc,test_loss=test(model,testloader,attack)
            tb.add_scalar('Train Loss',train_loss,epoch)
            tb.add_scalar('Accuracy on Trainset',present_trainset_acc,epoch)
            tb.add_scalar('Test Loss',test_loss,epoch)
            tb.add_scalar('Test_Adv_Acc',abs(adv_acc),epoch)
            tb.add_scalar('Accuracy on Testset',present_testset_acc,epoch)
            scheduler.step()
            if present_testset_acc>best_acc:
                state={
                    'network':model.state_dict(),
                    'accuracy':present_testset_acc,
                    'optimizer':optimizer.state_dict(),
                    'epoch':epoch
                }
                torch.save(state,f'./state_dicts/stdtrain/{args.learning_model}-{dataset}-{args.seed}-{args.task}.pkl')
                best_acc=present_testset_acc
            print("epoch",epoch,"loss",train_loss,"Train acc",present_trainset_acc ,"Test acc ",present_testset_acc)
        tb.close()
        return best_acc
    # model.load_state_dict(torch.load('/home/huan1932/TransRobust/res50cifar100.pkl')["network"])
    
    # test(model,test_loader)
    opt = torch.optim.SGD(model.parameters(), args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    num_samples = len(train_dataset)

    # Get the shape of a single sample
    single_sample_shape, _= train_dataset[0]

    # Combine to infer overall shape
    overall_shape =  (num_samples,) + single_sample_shape.shape
    classifier = generate_atk(model,criterion=criterion,optimizer=opt,min=minv,max=maxv,shape= overall_shape,number=numofclass)
    # attack = ProjectedGradientDescent(estimator=classifier,norm=args.norm, eps=args.eps,eps_step=args.eps_step,targeted = args.target,max_iter= args.max_iter)
    attack = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(),
                                eps=args.eps, nb_iter=args.max_iter, eps_iter=args.eps_step, rand_init=True, clip_min=0., clip_max=1., targeted = args.target)
    
    #-----20240116-----
    bestacc = train(attack)
    print('The best Acc is ',bestacc)
    # model.load_state_dict(torch.load(f'./state_dicts/stdtrain/{args.learning_model}-{dataset}-{args.seed}-{args.task}.pkl')['network'])
    # test(model,test_loader,attack)
    
elif args.Mode == "adversary":
    class Trainer():
        def __init__(self, args, attack):
            self.args = args
            self.attack = attack

        def train(self, model, tr_loader, va_loader=None):
            args = self.args
            model = model.to(device)
            opt = torch.optim.SGD(model.parameters(), args.lr, 
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
            iter_per_epoch = len(tr_loader) 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch)
            _iter = 0
            test_acc_track = []
            adv_test_acc_track = []
            begin_time = time.time()
            best_va_adv_acc = 0.0
            comment=f'{args.learning_model} with {args.task_name} on {numofclass} batch_size={args.batch} lr={args.lr} device={device}'
            tb=SummaryWriter(comment=comment)
            
            for epoch in range(1, args.epoch+1):
                batch_num=0
                train_correct=0
                train_loss = 0
                for data, label in tr_loader:
                    data, label = data.to(device), label.to(device)
                    data_time.update(time.time() - end)
                    model.train()
                    #--------maggie add--------
                    if args.datatype == 'network':
                        data = data.unsqueeze(1)
                        
                        data = data.unsqueeze(3)
                      
                    """ 
                    data.shape: torch.Size([1024, 53])
                    data.shape: torch.Size([1024, 1, 53])
                    data.shape: torch.Size([1024, 1, 53, 1])
                    """
                    #---------------------------
                    adv = torch.rand_like(data)
                    batch_size = len(data)
                    if args.target != False:
                        a = attack.perturb(data[label == 0], y=torch.randint(1,numofclass).item())
                        b = attack.perturb(data[label != 0], y=0)
                        adv[label == 0] = a
                        adv[label != 0] = b
                        adv = adv.cuda()
                    else:
                        adv = attack.perturb(data).cuda()
                    # if args.target == True:
                    #     adv_data[label!=0] = attack.generate(data[label!=0].cpu().numpy(), y=np.zeros((data[label!=0].shape[0],)))
                    #     adv_data[label==0] = attack.generate(data[label==0].cpu().numpy(), y=torch.randint(1,numofclass,size = (data[label==0].shape[0],)))
                    #     adv_data = torch.from_numpy(adv_data).float().to(device)
                    # else:
                    #     adv_data = self.attack.generate(data.cpu().numpy())
                    #     adv_data = torch.from_numpy(adv_data).float().to(device)
                    output = model(adv)
                    ogoutput = model(data)
                    ogloss = F.cross_entropy(ogoutput,label)
                    loss = F.cross_entropy(output, label)
                    loss = loss + ogloss*args.lambda_pgdat
                    train_loss += loss.item()
                    train_correct+=ogoutput.argmax(dim=1).eq(label).sum().item()
            
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if _iter % args.n_eval_step == 0:
                        t1 = time.time()
                        with torch.no_grad():
                            model.eval()
                            
                            stand_output = model(data)
                            adv_output = model(adv)
                            
                            model.train()
                        pred = torch.max(stand_output, dim=1)[1]
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                        pred = torch.max(adv_output, dim=1)[1]
                        
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        t2 = time.time()

                        begin_time = time.time()
                        print("epoch",epoch,"batch",batch_num,"adv_train_acc",adv_acc,"std_train_acc",std_acc)
                    present_trainset_acc=train_correct/len(train_dataset)
                    _iter += 1
                    scheduler.step()
                    batch_num=batch_num+1
                tb.add_scalar('Train Loss',train_loss,epoch)
                tb.add_scalar('Accuracy on Trainset',present_trainset_acc,epoch)
                
                
                if va_loader is not None:
                    t1 = time.time()
                    if not args.is_binary:
                        va_acc, va_adv_acc, test_loss= self.test(model, va_loader, False)
                    else: 
                        va_acc, va_adv_acc,_ = self.test(model, va_loader, False)
                    va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                    t2 = time.time()
                    if va_adv_acc > best_va_adv_acc:
                        best_va_adv_acc = va_adv_acc
                        if not os.path.exists(f"result/AdvTrain/PGD-AT/{dataset}-widene34-10"):
                            os.makedirs(f"result/AdvTrain/PGD-AT/{dataset}-widene34-10")
                        file_name = os.path.join(f"result/AdvTrain/PGD-AT/{dataset}-widene34-10", f'checkpointinf_best{args.learning_model}_inf{args.norm}_eps{args.eps}_eps_step{args.eps_step}_maxiter{args.max_iter}_{args.how}_{args.lambda_pgdat}_v2.pth')
                        torch.save(model.state_dict(),file_name)
                    test_acc_track.append(va_acc)
                    adv_test_acc_track.append(va_adv_acc)
                    print("adv test acc",va_adv_acc,"test acc",va_acc)
                    pickle.dump(test_acc_track,open(args.model_folder+'/test_acc_track.pkl','wb'))
                    pickle.dump(adv_test_acc_track,open(args.model_folder+'/adv_test_acc_track.pkl','wb'))
                if not os.path.exists("./result"):
                    os.mkdir("./result")
                tb.add_scalar('Test Loss',test_loss,epoch)
                tb.add_scalar('Accuracy on Testset',va_acc,epoch)
                file_name = os.path.join(f"result/AdvTrain/PGD-AT/{dataset}-widene34-10/", f'checkpointinf{args.learning_model}_inf{args.norm}_eps{args.eps}_eps_step{args.eps_step}_maxiter{args.max_iter}_{args.how}_{args.lambda_pgdat}_v2.pth')
                torch.save(model.state_dict(), file_name)
            tb.close()
            if not os.path.exists("./result"):
                os.mkdir("./result")
            file_name = os.path.join(f"result/AdvTrain/PGD-AT/{dataset}-widene34-10/", f'checkpointinf_final{args.learning_model}_inf{args.norm}_eps{args.eps}_eps_step{args.eps_step}_maxiter{args.max_iter}_{args.how}_{args.lambda_pgdat}.pth')
            torch.save(model.state_dict(), file_name)

        def test(self, model, loader, use_pseudo_label=False):

            total_acc = 0.0
            totalf1 = 0.0
            totalrecall = 0.0
            totalprecision = 0.0
            fnr = 0.0
            fpr= 0.0
            num = 0
            batchnum = 0
            total_adv_acc = 0.0
            totalf1adv = 0.0
            totalrecalladv = 0.0
            totalprecisionadv = 0.0
            fnradv = 0.0
            fpradv= 0.0
            total_loss = 0
            
            model.eval()
            with torch.no_grad():
                for data, label in loader:
                    data, label = data.to(device), label.to(device)
                    if args.datatype == 'network':
                        data = data.unsqueeze(1)
                        
                        data = data.unsqueeze(3)
                    output = model(data)

                    pred = torch.max(output, dim=1)[1]
                    te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    if args.is_binary:
                        totalf1 += f1_score1(y_true=label.cpu().numpy(),y_pred=pred.cpu().numpy())
                        totalprecision+=precision_score1(label.cpu().numpy(),pred.cpu().numpy())
                        totalrecall +=recall_score1(label.cpu().numpy(),pred.cpu().numpy())
                        fp,fn = fpandfn1(pred.cpu().numpy(), label.cpu().numpy())
                        fnr+= fp
                        fpr+= fn
                    total_acc += te_acc
                    num += output.shape[0]
                    adv = torch.rand_like(data)
                    # use predicted label as target label
                    with torch.enable_grad():
                        if args.target != False:
                            a = attack.perturb(data[label == 0], y=torch.randint(1,numofclass).item())
                            b = attack.perturb(data[label != 0], y=0)
                            adv[label == 0] = a
                            adv[label != 0] = b
                        else:
                            adv = attack.perturb(data)
                        # if args.target == True:
                        #     adv_data[label!=0] = attack.generate(data[label!=0].cpu().numpy(), y=np.zeros((data[label!=0].shape[0],)))
                        #     adv_data[label==0] = attack.generate(data[label==0].cpu().numpy(), y=torch.randint(1,numofclass,size = (data[label==0].shape[0],)))
                        #     adv_data = torch.from_numpy(adv_data).float().to(device)
                        # else:
                        #     adv_data = self.attack.generate(data.cpu().numpy())
                        #     adv_data = torch.from_numpy(adv_data).float().to(device)
                    model.eval()
                    adv_output = model(adv)
                    ogloss = F.cross_entropy(output,label)
                    loss = F.cross_entropy(adv_output, label)
                    loss = loss + ogloss*args.lambda_pgdat
                    total_loss += loss.item()
                    #print(adv_output.size())
                    #print(adv_output[0,:])
                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    
                    if args.is_binary:
                        totalf1adv += f1_score1(label.cpu().numpy(),adv_pred.cpu().numpy())
                        totalprecisionadv+=precision_score1(label.cpu().numpy(),adv_pred.cpu().numpy())
                        totalrecalladv +=recall_score1(label.cpu().numpy(),adv_pred.cpu().numpy())
                        fp,fn = fpandfn1(adv_pred.cpu().numpy(), label.cpu().numpy())
                        fnradv+= fp
                        fpradv+= fn
                    total_adv_acc += adv_acc
                    batchnum+=1
            model.train()
            if args.is_binary:
                all = [totalf1/batchnum,totalf1adv/batchnum,totalprecision/batchnum,totalprecisionadv/batchnum,totalrecall/batchnum,totalrecalladv/batchnum,fpr/batchnum,fpradv/batchnum,fnr/batchnum,fnradv/batchnum]
                print(total_acc / num , total_adv_acc / num)
                print(f"F1:{all[0]}, F1adv:{all[1]}, Precision:{all[2]}, Precisionadv:{all[3]},Recall:{all[4]}, recalladv:{all[5]},FPR:{all[6]}, FPRadv:{all[7]},FNR: {all[8]}, FNRadv{all[9]}")
                return total_acc / num , total_adv_acc / num,all
            else:
                print(total_acc / num , total_adv_acc / num)
                return total_acc / num , total_adv_acc / num,total_loss
    opt = torch.optim.SGD(model.parameters(), args.lr, 
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    num_samples = len(train_dataset)

    # Get the shape of a single sample
    single_sample_shape, _ = train_dataset[0]

    # Combine to infer overall shape
    overall_shape = (single_sample_shape.shape[0],)
    print(minv,maxv)
    classifier = generate_atk(model,criterion=criterion,optimizer=opt,min=minv,max=maxv,shape= overall_shape,number=numofclass)
    attack = ProjectedGradientDescent(estimator=classifier,norm=args.norm, eps=args.eps,eps_step=args.eps_step,targeted = args.target,max_iter= args.max_iter)
    # x_test_adv = attack.generate(x=cleanx,y=np.zeros((cleanx.shape[0],))) 
    attack = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(),
                                eps=args.eps, nb_iter=args.max_iter, eps_iter=args.eps_step, rand_init=True, clip_min=0., clip_max=1., targeted = args.target)
    trainer = Trainer(args,attack)  
    trainer.train(model, train_loader, test_loader)
    # model.load_state_dict(torch.load("result/AdvTrain/PGD-AT/UNSW-9classes-widene34-10/checkpointinf_bestwide_infinf_eps0.1_eps_step0.05_maxiter20_b.pth"))
    trainer.test(model,test_loader)

elif args.Mode == "trade":
    class Trainer():
        def __init__(self, args, attack):
            self.args = args
            self.attack = attack


        def train(self, model, tr_loader, va_loader=None):
            args = self.args
            if args.dataset in ['data/UNSW-NB15/train.csv','data/UNSW-NB15/traineandb.csv','data/UNSW-NB15/train.csv']:
                dataset = 'UNSW'
            elif args.dataset in ['cifar10','cifar100']:
                dataset = args.dataset
            elif args.dataset in ['data/NSL-KDD/kddbinarytrain.csv','data/NSL-KDD/kddmultitrain.csv']:
                dataset = 'KDD'
            opt = torch.optim.SGD(model.parameters(), args.lr, 
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum)
            iter_per_epoch = len(tr_loader) 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch)
            _iter = 0
            test_acc_track = []
            adv_test_acc_track = []
            begin_time = time.time()
            criterion_kl = nn.KLDivLoss(size_average=False)
            best_va_adv_acc = 0.0
            for epoch in range(1, args.epoch+1):
                for data, label in tr_loader:
                    
                    data, label = data.to(device), label.to(device)
                    data_time.update(time.time() - end)
                    if args.datatype == 'network':
                        data = data.unsqueeze(1)
                        
                        data = data.unsqueeze(3)
                    model.train()
                    batch_size = len(data)
                    
                    model.train()
                    adv_data = torch.rand_like(data)
                    if args.target == True:
                        adv_data[label!=0] = attack.generate(data[label!=0].cpu().numpy(), y=np.zeros((data[label!=0].shape[0],)))
                        adv_data[label==0] = attack.generate(data[label==0].cpu().numpy(), y=torch.randint(1,numofclass,size = (data[label==0].shape[0],)))
                        adv_data = torch.from_numpy(adv_data).float().to(device)
                    else:
                        adv_data = self.attack.generate(data.cpu().numpy())
                        adv_data = torch.from_numpy(adv_data).float().to(device)    
                    output = model(adv_data)
                    
                    
                    output_nature = model(data)
                    loss = F.cross_entropy(output_nature, label)
                    #loss_IN = F.cross_entropy(output_IN, label_IN)
                    loss_TRADES_robust = (1.0 / args.batch) * criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_nature, dim=1))
                    opt.zero_grad()
                    (loss + args.beta_trades * loss_TRADES_robust).backward()

                    opt.step()

                    if _iter % args.n_eval_step == 0:
                        t1 = time.time()

                        
                        with torch.no_grad():
                            model.eval()
                            stand_output = model(data)
                            model.train()
                        pred = torch.max(stand_output, dim=1)[1]

                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        

                        print("standard accuracy",std_acc,"adv accuracy",adv_acc,"epoch",epoch)

                        

                        

                    _iter += 1
                    scheduler.step()
                if not os.path.exists(f"result/AdvTrain/TRADES-PGD-AT/{dataset}-widene34-10/"):
                    os.makedirs(f"result/AdvTrain/TRADES-PGD-AT/{dataset}-widene34-10/")
                if va_loader is not None:
                    t1 = time.time()
                    va_acc, va_adv_acc = self.test(model, va_loader, False)
                    va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                    t2 = time.time()
                    if va_adv_acc > best_va_adv_acc:
                        best_va_adv_acc = va_adv_acc
                        file_name = os.path.join(f"result/AdvTrain/TRADES-PGD-AT/{dataset}-widene34-10/", f'checkpointtrade_bestinf{args.learning_model}_inf{args.norm}_eps{args.eps}_eps_step{args.eps_step}_maxiter{args.max_iter}_{args.how}_{args.lambda_pgdat}.pth')
                        torch.save(model.state_dict(), file_name)
                    test_acc_track.append(va_acc)
                    adv_test_acc_track.append(va_adv_acc)
                    pickle.dump(test_acc_track,open(args.model_folder+'/test_acc_tracktrade.pkl','wb'))
                    pickle.dump(adv_test_acc_track,open(args.model_folder+'/adv_test_acc_tracktrade.pkl','wb'))
            file_name = os.path.join(f"result/AdvTrain/TRADES-PGD-AT/{dataset}-widene34-10/", f'checkpointtrade_finalinf{args.learning_model}_inf{args.norm}_eps{args.eps}_eps_step{args.eps_step}_maxiter{args.max_iter}_{args.how}_{args.lambda_pgdat}.pth')
            torch.save(model.state_dict(), file_name)

        def test(self, model, loader, use_pseudo_label=False):
            # adv_test is False, return adv_acc as -1 

            total_acc = 0.0
            totalf1 = 0.0
            totalrecall = 0.0
            totalprecision = 0.0
            fnr = 0.0
            fpr= 0.0
            num = 0
            batchnum = 0
            total_adv_acc = 0.0
            totalf1adv = 0.0
            totalrecalladv = 0.0
            totalprecisionadv = 0.0
            fnradv = 0.0
            fpradv= 0.0
            model.eval()
            with torch.no_grad():
                for data, label in loader:
                    data, label = data.to(device), label.to(device)
                    if args.datatype == 'network':
                        data = data.unsqueeze(1)
                        
                        data = data.unsqueeze(3)
                    output = model(data)

                    pred = torch.max(output, dim=1)[1]
                    te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    if args.is_binary:
                        totalf1 += f1_score1(label.cpu().numpy(),pred.cpu().numpy())
                        totalprecision+=precision_score1(label.cpu().numpy(),pred.cpu().numpy())
                        totalrecall +=recall_score1(label.cpu().numpy(),pred.cpu().numpy())
                        fp,fn = fpandfn1(pred.cpu().numpy(), label.cpu().numpy())
                        fnr+= fp
                        fpr+= fn
                    total_acc += te_acc
                    num += output.shape[0]
                    adv_data = torch.rand_like(data)
                    with torch.enable_grad():
                    # use predicted label as target label
                        if args.target == True:
                            adv_data[label!=0] = attack.generate(data[label!=0].cpu().numpy(), y=np.zeros((data[label!=0].shape[0],)))
                            adv_data[label==0] = attack.generate(data[label==0].cpu().numpy(), y=torch.randint(1,numofclass,size = (data[label==0].shape[0],)))
                            adv_data = torch.from_numpy(adv_data).float().to(device)
                        else:
                            adv_data = self.attack.generate(data.cpu().numpy())
                            adv_data = torch.from_numpy(adv_data).float().to(device)
                
                    model.eval()
                    adv_output = model(adv_data)
                    #print(adv_output.size())
                    #print(adv_output[0,:])
                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    if args.is_binary:
                        totalf1adv += f1_score1(label.cpu().numpy(),adv_pred.cpu().numpy())
                        totalprecisionadv+=precision_score1(label.cpu().numpy(),adv_pred.cpu().numpy())
                        totalrecalladv +=recall_score1(label.cpu().numpy(),adv_pred.cpu().numpy())
                        fp,fn = fpandfn1(adv_pred.cpu().numpy(), label.cpu().numpy())
                        fnradv+= fp
                        fpradv+= fn
                    total_adv_acc += adv_acc
                    batchnum+=1
            model.train()
            if args.is_binary:
                all = [totalf1/batchnum,totalf1adv/batchnum,totalprecision/batchnum,totalprecisionadv/batchnum,totalrecall/batchnum,totalrecalladv/batchnum,fpr/batchnum,fpradv/batchnum,fnr/batchnum,fnradv/batchnum]
                print(total_acc / num , total_adv_acc / num)
                print(f"F1:{all[0]}, F1adv:{all[1]}, Precision:{all[2]}, Precisionadv:{all[3]},Recall:{all[4]}, recalladv:{all[5]},FPR:{all[6]}, FPRadv:{all[7]},FNR: {all[8]}, FNRadv{all[9]}")
                return total_acc / num , total_adv_acc / num
            else:
                print(total_acc / num , total_adv_acc / num)
                return total_acc / num , total_adv_acc / num
    
    opt = torch.optim.SGD(model.parameters(), args.lr, 
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    num_samples = len(train_dataset)

    # Get the shape of a single sample
    single_sample_shape, _ = train_dataset[0]

    # Combine to infer overall shape
    overall_shape = (num_samples,) + single_sample_shape.shape
    classifier = generate_atk(model,criterion=criterion,optimizer=opt,min=minv,max=maxv,shape= overall_shape,number=numofclass)
    attack = ProjectedGradientDescent(estimator=classifier,norm=args.norm, eps=args.eps,eps_step=args.eps_step,targeted = args.target,max_iter= args.max_iter)
    # x_test_adv = attack.generate(x=cleanx,y=np.zeros((cleanx.shape[0],))) 
    trainer = Trainer(args,attack)  
    trainer.train(model, train_loader, test_loader)
    # model.load_state_dict(torch.load("./result_trade/checkpointtrade_bestinfcifar100for100.pth"))
    trainer.test(model,test_loader)
elif args.Mode == "finetune-twins":
    class Trainer():
        def __init__(self, args, attack):
            self.args = args
            
            self.attack = attack
            

        def train(self, model, tr_loader, va_loader=None, adv_train=True):
            args = self.args

            opt = torch.optim.SGD(model.parameters(), args.lr, 
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
            iter_per_epoch = len(tr_loader) 
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                            milestones=[iter_per_epoch*30, iter_per_epoch*50], 
                                                            gamma=0.1)
            _iter = 0
            test_acc_track = []
            adv_test_acc_track = []
           
            begin_time = time.time()
           
            best_va_adv_acc = 0.0
            for epoch in range(1, args.epoch+1):
                for data, label in tr_loader:
                    
                    data, label = data.to(device), label.to(device)
                    data_time.update(time.time() - end)
                    if args.datatype == 'network':
                        data = data.unsqueeze(1)
                        
                        data = data.unsqueeze(3)
                    model.train()
                    batch_size = len(data)
                    adv_data = torch.rand_like(data)
                    if adv_train:
                        
                    
                        if args.target == True:
                            adv_data[label!=0] = self.attack.generate(data[label!=0].cpu().numpy(), y=np.zeros((data[label!=0].shape[0],)))
                            adv_data[label==0] = self.attack.generate(data[label==0].cpu().numpy(), y=torch.randint(1,numofclass,size = (data[label==0].shape[0],)))
                            adv_data = torch.from_numpy(adv_data).float().to(device)
                        else:
                            adv_data = self.attack.generate(data.cpu().numpy())
                            adv_data = torch.from_numpy(adv_data).float().to(device)
                        images = torch.cat([adv_data[:batch_size // 2], adv_data[batch_size // 2:]], dim=0)
                        targets = torch.cat([label[:batch_size // 2], label[batch_size // 2:]], dim=0)
                        if len(targets) % 2 !=0:
                            images = torch.cat([adv_data[:batch_size // 2], adv_data[batch_size // 2:-1]], dim=0)
                            targets = torch.cat([label[:batch_size // 2], label[batch_size // 2:-1]], dim=0)
                        outputs = model(images)
                    else:
                        outputs = model(data)
                    targets = targets.view(2, batch_size//2).transpose(1, 0)
                    outputs = outputs.transpose(1, 0).contiguous().view(-1, numofclass)
                    targets = targets.transpose(1, 0).contiguous().view(-1)
                    loss_main = F.cross_entropy(outputs[:batch_size // 2], targets[:batch_size // 2])
                    loss_aux = F.cross_entropy(outputs[batch_size // 2:], targets[batch_size // 2:])
                    loss = args.lambda_twins * loss_main + loss_aux
                    opt.zero_grad()
                    loss.backward()
                    
                    opt.step()
                    
                    prec1_main = accuracy(outputs.data[:batch_size // 2],
                                        targets.data[:batch_size // 2], topk=(1,))[0]
                    prec1_aux  = accuracy(outputs.data[batch_size // 2:],
                                        targets.data[batch_size // 2:], topk=(1,))[0]
                    
                    print('fixed bn acc: '+str(prec1_main.cpu().detach().numpy())+', adaptive bn acc:'+str(prec1_aux.cpu().detach().numpy()))
                    if _iter % args.n_eval_step == 0:
                        t1 = time.time()
                        adv_data= torch.rand_like(data)
                        if adv_train:
                            with torch.no_grad():
                                model.eval()
                                stand_output = model(data)
                                model.train()
                            pred = torch.max(stand_output, dim=1)[1]

                            # print(pred)
                            std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                            pred = torch.max(outputs, dim=1)[1]
                            # print(pred)
                            adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        else:
                            if args.target == True:
                                adv_data[label!=0] = self.attack.generate(data[label!=0].cpu().numpy(), y=np.zeros((data[label!=0].shape[0],)))
                                adv_data[label==0] = self.attack.generate(data[label==0].cpu().numpy(), y=torch.randint(1,numofclass,size = (data[label==0].shape[0],)))
                                adv_data = torch.from_numpy(adv_data).float().to(device)
                            else:
                                adv_data = self.attack.generate(data.cpu().numpy())
                                adv_data = torch.from_numpy(adv_data).float().to(device)
                            with torch.no_grad():
                                model.eval()
                                adv_output = model(adv_data)
                                model.train()
                            pred = torch.max(adv_output, dim=1)[1]
                            # print(label)
                            # print(pred)
                            adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                            pred = torch.max(outputs, dim=1)[1]
                            # print(pred)
                            std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        t2 = time.time()


                        print("Standard:",std_acc,"adv:",adv_acc)

                        begin_time = time.time()

                   

                    _iter += 1
                    
                    scheduler.step()

                if va_loader is not None:
                    t1 = time.time()
                    if args.is_binary:
                        va_acc, va_adv_acc,_ = self.test(model, va_loader, True)
                        va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0
                    else:
                        va_acc, va_adv_acc = self.test(model, va_loader, True)
                        va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                    t2 = time.time()
                    
                    if va_adv_acc > best_va_adv_acc:
                        best_va_adv_acc = va_adv_acc
                        file_name = os.path.join('./result/transfer/Twins/', f'checkpoint_besttwins.pth')
                        torch.save(model, file_name)
                    
                    if epoch % args.n_eval_step == 0:
                        test_acc_track.append(va_acc)
                        adv_test_acc_track.append(va_adv_acc)
                        pickle.dump(test_acc_track,open(args.model_folder+'/test_acc_tracktwins.pkl','wb'))
                        pickle.dump(adv_test_acc_track,open(args.model_folder+'/adv_test_acc_tracktwins.pkl','wb'))
            file_name = os.path.join('./result/transfer/Twins/', f'checkpoint_finaltwins.pth')
            torch.save(model.state_dict(), file_name)

        def test(self, model, loader, adv_test=False):
            # adv_test is False, return adv_acc as -1 
            totaln_acc = 0.0
            total_acc = 0.0
            totalf1 = 0.0
            totalrecall = 0.0
            totalprecision = 0.0
            fnr = 0.0
            fpr= 0.0
            num = 0
            batchnum = 0
            total_adv_acc = 0.0
            totalf1adv = 0.0
            totalrecalladv = 0.0
            totalprecisionadv = 0.0
            fnradv = 0.0
            fpradv= 0.0
            model.eval()
            with torch.no_grad():
                for data, label in loader:
                    data, label = data.to(device), label.to(device)
                    if args.datatype == 'network':
                        data = data.unsqueeze(1)
                        
                        data = data.unsqueeze(3)
                    output = model(data)
                    noise = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=data.shape), dtype=torch.float32).to(data.device)
                    outputn = model(data+noise)
                    pred = torch.max(output, dim=1)[1]
                    predn = torch.max(outputn,dim=1)[1]
                    te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    n = evaluate(predn.cpu().numpy(), label.cpu().numpy(), 'sum')
                    if args.is_binary:
                        totalf1 += f1_score1(label.cpu().numpy(),pred.cpu().numpy())
                        totalprecision+=precision_score1(label.cpu().numpy(),pred.cpu().numpy())
                        totalrecall +=recall_score1(label.cpu().numpy(),pred.cpu().numpy())
                        fp,fn = fpandfn1(pred.cpu().numpy(), label.cpu().numpy())
                        fnr+= fp
                        fpr+= fn
                    total_acc += te_acc
                    totaln_acc += n
                    num += output.shape[0]

                    if adv_test:
                        # use predicted label as target label
                        adv_data = torch.rand_like(data)
                        with torch.enable_grad():
                            if args.target == True:
                                adv_data[label!=0] = attack.generate(data[label!=0].cpu().numpy(), y=np.zeros((data[label!=0].shape[0],)))
                                adv_data[label==0] = attack.generate(data[label==0].cpu().numpy(), y=torch.randint(1,numofclass,size = (data[label==0].shape[0],)))
                                adv_data = torch.from_numpy(adv_data).float().to(device)
                            else:
                                adv_data = self.attack.generate(data.cpu().numpy())
                                adv_data = torch.from_numpy(adv_data).float().to(device)
                        model.eval()
                        adv_output = model(adv_data)
                        #print(adv_output.size())
                        #print(adv_output[0,:])
                        adv_pred = torch.max(adv_output, dim=1)[1]
                        adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                        total_adv_acc += adv_acc
                        if args.is_binary:
                            totalf1adv += f1_score1(label.cpu().numpy(),adv_pred.cpu().numpy())
                            totalprecisionadv+=precision_score1(label.cpu().numpy(),adv_pred.cpu().numpy())
                            totalrecalladv +=recall_score1(label.cpu().numpy(),adv_pred.cpu().numpy())
                            fp,fn = fpandfn1(adv_pred.cpu().numpy(), label.cpu().numpy())
                            fnradv+= fp
                            fpradv+= fn
                        batchnum+=1
                    else:
                        total_adv_acc = -num
            model.train()
            if args.is_binary:
                all = [totalf1/batchnum,totalf1adv/batchnum,totalprecision/batchnum,totalprecisionadv/batchnum,totalrecall/batchnum,totalrecalladv/batchnum,fpr/batchnum,fpradv/batchnum,fnr/batchnum,fnradv/batchnum]
                print(total_acc / num , total_adv_acc / num, totaln_acc / num)
                print(f"F1:{all[0]}, F1adv:{all[1]}, Precision:{all[2]}, Precisionadv:{all[3]},Recall:{all[4]}, recalladv:{all[5]},FPR:{all[6]}, FPRadv:{all[7]},FNR: {all[8]}, FNRadv{all[9]}")
                return total_acc / num , total_adv_acc / num,all
            else:
                print(total_acc / num , total_adv_acc / num,totaln_acc / num)
                return total_acc / num , total_adv_acc / num

    #model = resnet.ResNet18()  
    # if args.teacher_model == 'res':
    #     model_arch = "resnet18"
    # elif args.teacher_model =='wide':
    #     model_arch = 'widenet'
    # if args.twins_source in ['unsw']:
    #     dataset = datasets.UNSW('',mean = torch.tensor([0]), std = torch.tensor([1]),num_classes = args.twins_class)
    # elif args.twins_source in 'cifar100':
    #     dataset = datasets.cifar100('')
    # elif args.twins_source in ['kdd']:
    #     dataset = datasets.KDD('',mean = torch.tensor([0]), std = torch.tensor([1]),num_classes = args.twins_class)  
    # model, _ = model_utils.make_and_restore_model(
    #             model=model,
    #             dataset=dataset, resume_path=args.pretrained,add_custom_forward=True)
    
    # while hasattr(model, 'model'):
    #     model = model.model
    # model = fine_tunify.ft(
    #             model_arch, model, numofclass, 0)
    # if args.dataset in ['data/UNSW-NB15/train.csv','data/UNSW-NB15/traineandb.csv','data/UNSW-NB15/train.csv']:
    #     dataset = datasets.UNSW('',mean = torch.tensor([0]), std = torch.tensor([1]))
    #     data = datasets.UNSW()
    # elif args.dataset in 'cifar10':
    #     dataset = datasets.cifar10('')
    #     data = datasets.CIFAR()
    # elif args.dataset in ['data/NSL-KDD/kddbinarytrain.csv','data/NSL-KDD/kddmultitrain.csv']:
    #     dataset = datasets.KDD('',mean = torch.tensor([0]), std = torch.tensor([1]))
    #     data  = datasets.KDD()
    # ds, (_,_) = transfer_datasets.make_loaders(dataset, batch_size=128, workers=8, subset=50000)
    # print(ds)
    # if type(ds) == int:
    #     print('new ds')
    #     new_ds = datasets.CIFAR()
    #     new_ds.num_classes = ds
    #     if args.datatype =='image':
    #         new_ds.mean = torch.tensor([0., 0., 0.])
    #         new_ds.std = torch.tensor([1.0, 1.0, 1.0])
    #     else: 
    #         new_ds.mean = torch.tensor([0])
    #         new_ds.std = torch.tensor([1])
    #     ds = new_ds
    # ds.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    # ds.std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    # model, checkpoint = model_utils.make_and_restore_model(model=model, dataset=ds, add_custom_forward=True)
    def load_teacher(path,model,args):
        state_dict = torch.load(path)
        
        number = state_dict["fc.weight"].shape[0]
        if model == 'wide':
            teacher_net = WideResNet(num_classes=number,depth=args.depth,widen_factor=args.widening,args=args)
        else:
            raise Exception
        teacher_net.load_state_dict(state_dict)
        return teacher_net
    additional_hidden = 0
    model = load_teacher(args.pretrained,args.teacher_model, args)
    if args.teacher_model == 'res':
        num_ftrs = model.linear.in_features
        # The two cases are split just to allow loading
        # models trained prior to adding the additional_hidden argument
        # without errors
        if additional_hidden == 0:
            model.linear = nn.Linear(num_ftrs, numofclass)
        else:
            model.linear = SequentialWithArgs(
                *list(sum([[nn.Linear(num_ftrs, num_ftrs), nn.ReLU()] for i in range(additional_hidden)], [])),
                nn.Linear(num_ftrs, numofclass)
            )
        input_size = 32
    elif args.teacher_model == 'wide':
        num_ftrs = model.fc.in_features
        # The two cases are split just to allow loading
        # models trained prior to adding the additional_hidden argument
        # without errors
        if additional_hidden == 0:
            model.fc = nn.Linear(num_ftrs, numofclass)
        else:
            model.fc = SequentialWithArgs(
                *list(sum([[nn.Linear(num_ftrs, num_ftrs), nn.ReLU()] for i in range(additional_hidden)], [])),
                nn.Linear(num_ftrs, numofclass)
            )
        input_size = 32
    else:
        raise ValueError("Invalid model type, exiting...")
    opt = torch.optim.SGD(model.parameters(), args.lr, 
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    num_samples = len(train_dataset)

    # Get the shape of a single sample
    single_sample_shape, _ = train_dataset[0]

    # Combine to infer overall shape
    overall_shape = (num_samples,) + single_sample_shape.shape
    classifier = generate_atk(model,criterion=criterion,optimizer=opt,min=minv,max=maxv,shape= overall_shape,number=numofclass)
    attack = ProjectedGradientDescent(estimator=classifier,norm=args.norm, eps=args.eps,eps_step=args.eps_step,targeted = args.target,max_iter= args.max_iter)

    trainer = Trainer(args, attack)
    trainer.train(model, train_loader, test_loader)

    load_checkpoint = os.path.join('./result/transfer/Twins/', f'checkpoint_besttwins.pth')
    checkpoint = torch.load(load_checkpoint)
    model.load_state_dict(checkpoint)
    if args.is_binary:
        std_acc, adv_acc,_ = trainer.test(model, test_loader, adv_test=True)
    else:
        std_acc, adv_acc = trainer.test(model, test_loader, adv_test=True)
    print(f"std acc: {std_acc * 100:.3f}%, adv_acc: {adv_acc * 100:.3f}%")


elif args.Mode == "finetune":
    opt = torch.optim.SGD(model.parameters(), args.lr, 
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    num_samples = len(train_dataset)

    # Get the shape of a single sample
    single_sample_shape, _ = train_dataset[0]

    # Combine to infer overall shape
    overall_shape = (num_samples,) + single_sample_shape.shape
    classifier = generate_atk(model,criterion=criterion,optimizer=opt,min=minv,max=maxv,shape= overall_shape,number=numofclass)
    attack = ProjectedGradientDescent(estimator=classifier,norm=args.norm, eps=args.eps,eps_step=args.eps_step,targeted = args.target,max_iter= args.max_iter)
    def my_eval(data, model, attack):
        total_acc = 0.0
        totaln = 0
        totalf1 = 0.0
        totalrecall = 0.0
        totalprecision = 0.0
        fnr = 0.0
        fpr= 0.0
        num = 0
        total_loss = 0
        total_loss_adv = 0
        batchnum = 0
        total_adv_acc = 0.0
        totalf1adv = 0.0
        totalrecalladv = 0.0
        totalprecisionadv = 0.0
        fnradv = 0.0
        fpradv= 0.0
        loss = nn.CrossEntropyLoss()
        for x,y in data:
            x = x.to(device)
            y = y.to(device)
            if args.datatype == 'network':
                x = x.unsqueeze(1)
                
                x = x.unsqueeze(3)
            output = model(x)
            noise = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=x.shape), dtype=torch.float32).to(x.device)
            outputn = model(x+noise)
            pred = torch.max(output, dim=1)[1]
            predn = torch.max(outputn, dim=1)[1]
            te_acc = evaluate(pred.cpu().numpy(), y.cpu().numpy(), 'sum')
            n = evaluate(predn.cpu().numpy(), y.cpu().numpy(), 'sum')
            total_acc += te_acc
            totaln += n
            if args.is_binary:
                totalf1 += f1_score1(y.cpu().numpy(),pred.cpu().numpy())
                totalprecision+=precision_score1(y.cpu().numpy(),pred.cpu().numpy())
                totalrecall +=recall_score1(y.cpu().numpy(),pred.cpu().numpy())
                fp,fn = fpandfn1(pred.cpu().numpy(), y.cpu().numpy())
                fnr+= fp
                fpr+= fn
            num += output.shape[0]
            x_batch_adv = torch.rand_like(x)
            with torch.enable_grad():
                if args.target == True:
                    x_batch_adv[y!=0] = attack.generate(x[y!=0].cpu().numpy(), y=np.zeros((x[y!=0].shape[0],)))
                    x_batch_adv[y==0] = attack.generate(x[y==0].cpu().numpy(), y=torch.randint(1,numofclass,size = (x[y==0].shape[0],)))
                    x_batch_adv = torch.from_numpy(x_batch_adv).float().to(device)
                else:
                    x_batch_adv = attack.generate(x.cpu().numpy())
                    x_batch_adv = torch.from_numpy(x_batch_adv).float().to(device)
            model.eval()
            advout = model(x_batch_adv)
            advout = model(x_batch_adv)
            adv_pred = torch.max(advout, dim=1)[1]
            adv_acc = evaluate(adv_pred.cpu().numpy(), y.cpu().numpy(), 'sum')
            total_adv_acc += adv_acc
            if args.is_binary:
                totalf1adv += f1_score1(y.cpu().numpy(),adv_pred.cpu().numpy())
                totalprecisionadv+=precision_score1(y.cpu().numpy(),adv_pred.cpu().numpy())
                totalrecalladv +=recall_score1(y.cpu().numpy(),adv_pred.cpu().numpy())
                fp,fn = fpandfn1(adv_pred.cpu().numpy(), y.cpu().numpy())
                fnradv+= fp
                fpradv+= fn
            total_loss += loss(output,y).item()
            total_loss_adv += loss(advout,y).item()
            batchnum+=1
        print("Accuracy",total_acc / num ,"Adv Accuracy", total_adv_acc / num, 'Natural Cor Accuracy', totaln/num ,"average loss",total_loss/ num, "average adv loss",total_loss_adv/ num)
        if args.is_binary:
            all = [totalf1/batchnum,totalf1adv/batchnum,totalprecision/batchnum,totalprecisionadv/batchnum,totalrecall/batchnum,totalrecalladv/batchnum,fpr/batchnum,fpradv/batchnum,fnr/batchnum,fnradv/batchnum]
            print(f"F1:{all[0]}, F1adv:{all[1]}, Precision:{all[2]}, Precisionadv:{all[3]},Recall:{all[4]}, recalladv:{all[5]},FPR:{all[6]}, FPRadv:{all[7]},FNR: {all[8]}, FNRadv{all[9]}")
            return total_acc / num , total_adv_acc / num,all
        else:
            return total_acc / num , total_adv_acc / num
    class DataLoader(object):
        def __init__(self,xs,ys,args):
            self.xs = xs
            self.n = xs.shape[0]
            self.ys = ys
            self.batch_start = 0
            self.cur_order = np.random.permutation(self.n)
            if args.dataset in ['data/UNSW-NB15/train.csv','data/UNSW-NB15/traineandb.csv','data/UNSW-NB15/train.csv']:
                dataset = 'UNSW'
            elif args.dataset in ['cifar10','cifar100']:
                dataset = args.dataset
            elif args.dataset in ['data/NSL-KDD/kddbinarytrain.csv','data/NSL-KDD/kddmultitrain.csv']:
                dataset = 'KDD'
            path_to_feat_reps = args.featurepath + f'/robust_{dataset}_feats.npy'
            if os.path.exists(path_to_feat_reps):
                self.features = np.load(path_to_feat_reps)
            else:
                self.features = None

        def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
            
            if not multiple_passes:
                actual_batch_size = min(batch_size, self.n - self.batch_start)
                if actual_batch_size <= 0:
                    raise ValueError('Pass through the dataset is complete.')
                batch_end = self.batch_start + actual_batch_size
                batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
                batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
                self.batch_start += actual_batch_size
                return batch_xs, batch_ys
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size < batch_size:
                if reshuffle_after_pass:
                    self.cur_order = np.random.permutation(self.n)
                self.batch_start = 0
            batch_end = self.batch_start + batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            batch_fr = self.features[self.cur_order[self.batch_start: batch_end], ...]
            self.batch_start += batch_size
            return batch_xs, batch_ys, batch_fr
    pretrained_dict = torch.load(args.pretrained)
    if args.learning_model == "res":
        train_vars_last = model.linear.parameters()
    else: 
        train_vars_last = model.fc.parameters()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    feats_dir = os.path.join(args.featurepath, npydir)
    if os.path.exists(feats_dir):
        print('the feature representations already exist ... moving on to training')
    else:

        print('saving feature representations')

        # Assuming 'dataset' is your PyTorch dataset for training data
        # and 'batch_size' is defined
        n_train = len(train_dataset)

        # Assuming 'model' is your PyTorch model and 'device' is your computation device (e.g., 'cuda' or 'cpu')
        model.to(device)
        model.eval()

        all_feats = []
        with torch.no_grad():
            for data,label in tqdm(train_loader):
                # Assuming data is a tuple of (features, labels)
                inputs = data.to(device)
                if args.datatype == 'network':
                    inputs = inputs.unsqueeze(1)
                    
                    inputs = inputs.unsqueeze(3) 
                # Get the feature representations from the model
                # Modify this line according to your model's method of feature extraction
                these_feats = model.featureextract(inputs)

                all_feats.append(these_feats.cpu().numpy())
        
        # Concatenate all features and reshape
        all_feats = np.vstack(all_feats)
        print(all_feats.shape)
        all_feats = all_feats.reshape(-1, 640)
        # Save the feature representations
        np.save(feats_dir, all_feats)

        print('saved all feat reps')
      
    # Initialize the model
    model.train()
    criterion = nn.CrossEntropyLoss()
    feat_sim_pen = args.feat_sim

    # Optimizer setup
    optimizer_all = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.learning_model == "res":
        optimizer_last = optim.SGD(model.linear.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer_last = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)
    # Learning rate scheduler
    scheduler_all = torch.optim.lr_scheduler.MultiStepLR(optimizer_all, milestones=[100], gamma=1)
    scheduler_last = torch.optim.lr_scheduler.MultiStepLR(optimizer_last, milestones=[100], gamma=1)


    # Checkpoint and logging setup
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    Xs = []
    Ys = []

    for image, label in train_dataset:
        if args.datatype == 'network':
            image = image.to(device)
            label = label.to(device)
        Xs.append(image)
        Ys.append(label)

    # Convert lists to tensors
    Xs = torch.stack(Xs)
    Ys = torch.tensor(Ys)
    # Training loop
    test_acc_track = []
    adv_test_acc_track = []
    best = 0
    for epoch in range(args.epoch):
        dataloader = DataLoader(Xs,Ys,args)
        for i in range(len(train_loader)):
            x_batch, y_batch, ft_batch = dataloader.get_next_batch(batch_size=args.batch,multiple_passes=True)
            data_time.update(time.time() - end)
            if args.datatype == 'network':
                x_batch = x_batch.unsqueeze(1)
                
                x_batch = x_batch.unsqueeze(3)
            x_batch, y_batch, ft_batch = x_batch.to(device), y_batch.to(device), torch.from_numpy(ft_batch).to(device)
            outputs = model(x_batch)
            loss_xent = criterion(outputs, y_batch)
            if args.learning_model =='res':
                loss_feat_sim = feat_sim_pen * torch.mean(torch.norm(model.copy - ft_batch, p=1, dim=1))
            else:
                loss_feat_sim = feat_sim_pen * torch.mean(torch.norm(model.feature - ft_batch, p=1, dim=1))
            loss = loss_xent + loss_feat_sim

            optimizer = optimizer_last if epoch <= args.warmstart_step else optimizer_all
            scheduler = scheduler_last if epoch <= args.warmstart_step else scheduler_all

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Learning rate scheduling
        scheduler.step()

        # Logging and checkpointing
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')
            if args.is_binary:
                std,adv,_ = my_eval(data=test_loader, model=model, attack=attack)
                test_acc_track.append(std)
                adv_test_acc_track.append(adv)
                pickle.dump(test_acc_track,open(args.model_folder+'/test_acc_trackft.pkl','wb'))
                pickle.dump(adv_test_acc_track,open(args.model_folder+'/adv_test_acc_trackft.pkl','wb'))
            else: 
                std,adv = my_eval(data=test_loader, model=model, attack=attack)
                test_acc_track.append(std)
                adv_test_acc_track.append(adv)
                pickle.dump(test_acc_track,open(args.model_folder+'/test_acc_trackft.pkl','wb'))
                pickle.dump(adv_test_acc_track,open(args.model_folder+'/adv_test_acc_trackft.pkl','wb'))
        if (epoch+1) % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join('./result/transfer/finetune/', f'checkpointforstandardft_{epoch}.pt'))

    # Evaluation
    if args.is_binary:
        std,adv,_ = my_eval(data=test_loader, model=model, attack=attack)
    else: 
        std,adv = my_eval(data=test_loader, model=model, attack=attack)
    
elif args.Mode == "baselinefinetune":
    opt = torch.optim.SGD(model.parameters(), args.lr, 
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    num_samples = len(train_dataset)

    # Get the shape of a single sample
    single_sample_shape, _ = train_dataset[0]

    # Combine to infer overall shape
    overall_shape = (num_samples,) + single_sample_shape.shape
    classifier = generate_atk(model,criterion=criterion,optimizer=opt,min=minv,max=maxv,shape= overall_shape,number=numofclass)
    attack = ProjectedGradientDescent(estimator=classifier,norm=args.norm, eps=args.eps,eps_step=args.eps_step,targeted = args.target,max_iter= args.max_iter)
    def my_eval(data, model, attack):
        total_acc = 0.0
        totaln = 0
        totalf1 = 0.0
        totalrecall = 0.0
        totalprecision = 0.0
        fnr = 0.0
        fpr= 0.0
        num = 0
        total_loss = 0
        total_loss_adv = 0
        batchnum = 0
        total_adv_acc = 0.0
        totalf1adv = 0.0
        totalrecalladv = 0.0
        totalprecisionadv = 0.0
        fnradv = 0.0
        fpradv= 0.0
        model.eval()
        loss = nn.CrossEntropyLoss()
        for x,y in data:
            x = x.to(device)
            y = y.to(device)
            if args.datatype == 'network':
                x = x.unsqueeze(1)
                
                x = x.unsqueeze(3)
            output = model(x)
            noise = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=x.shape), dtype=torch.float32).to(x.device)
            outputn = model(x+noise)
            pred = torch.max(output, dim=1)[1]
            predn = torch.max(outputn, dim=1)[1]
            te_acc = evaluate(pred.cpu().numpy(), y.cpu().numpy(), 'sum')
            n = evaluate(predn.cpu().numpy(), y.cpu().numpy(), 'sum')
            total_acc += te_acc
            totaln += n
            if args.is_binary:
                totalf1 += f1_score1(y.cpu().numpy(),pred.cpu().numpy())
                totalprecision+=precision_score1(y.cpu().numpy(),pred.cpu().numpy())
                totalrecall +=recall_score1(y.cpu().numpy(),pred.cpu().numpy())
                fp,fn = fpandfn1(pred.cpu().numpy(), y.cpu().numpy())
                fnr+= fp
                fpr+= fn
            num += output.shape[0]
            x_batch_adv = torch.rand_like(x)
            with torch.enable_grad():
                if args.target == True:
                    x_batch_adv[y!=0] = attack.generate(x[y!=0].cpu().numpy(), y=np.zeros((x[y!=0].shape[0],)))
                    x_batch_adv[y==0] = attack.generate(x[y==0].cpu().numpy(), y=torch.randint(1,numofclass,size = (x[y==0].shape[0],)))
                    x_batch_adv = torch.from_numpy(x_batch_adv).float().to(device)
                else:
                    x_batch_adv = attack.generate(x.cpu().numpy())
                    x_batch_adv = torch.from_numpy(x_batch_adv).float().to(device)
            model.eval()
            advout = model(x_batch_adv)
            adv_pred = torch.max(advout, dim=1)[1]
            adv_acc = evaluate(adv_pred.cpu().numpy(), y.cpu().numpy(), 'sum')
            total_adv_acc += adv_acc
            if args.is_binary:
                totalf1adv += f1_score1(y.cpu().numpy(),adv_pred.cpu().numpy())
                totalprecisionadv+=precision_score1(y.cpu().numpy(),adv_pred.cpu().numpy())
                totalrecalladv +=recall_score1(y.cpu().numpy(),adv_pred.cpu().numpy())
                fp,fn = fpandfn1(adv_pred.cpu().numpy(), y.cpu().numpy())
                fnradv+= fp
                fpradv+= fn
            total_loss += loss(output,y).item()
            total_loss_adv += loss(advout,y).item()
            batchnum+=1
        print("Accuracy",total_acc / num ,"Adv Accuracy", total_adv_acc / num, 'Natural Cor Accuracy', totaln/num ,"average loss",total_loss/ num, "average adv loss",total_loss_adv/ num)
        if args.is_binary:
            all = [totalf1/batchnum,totalf1adv/batchnum,totalprecision/batchnum,totalprecisionadv/batchnum,totalrecall/batchnum,totalrecalladv/batchnum,fpr/batchnum,fpradv/batchnum,fnr/batchnum,fnradv/batchnum]
            print(f"F1:{all[0]}, F1adv:{all[1]}, Precision:{all[2]}, Precisionadv:{all[3]},Recall:{all[4]}, recalladv:{all[5]},FPR:{all[6]}, FPRadv:{all[7]},FNR: {all[8]}, FNRadv{all[9]}")
            return total_acc / num , total_adv_acc / num,all
        else:
            return total_acc / num , total_adv_acc / num
    
    pretrained_dict = torch.load(args.pretrained)
    if args.learning_model == "res":
        train_vars_last = model.linear.parameters()
    else: 
        train_vars_last = model.fc.parameters()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

        
      
    # Initialize the model
    model.train()
    criterion = nn.CrossEntropyLoss()

    # Optimizer setup
    if args.learning_model == "res":
        optimizer = optim.SGD(model.linear.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=1)


    # Checkpoint and logging setup
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Training loop
    test_acc_track = []
    adv_test_acc_track = []
    best = 0
    for epoch in range(args.epoch):
        for x_batch,y_batch in train_loader:
            if args.datatype == 'network':
                x_batch = x_batch.unsqueeze(1)
                
                x_batch = x_batch.unsqueeze(3)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            data_time.update(time.time() - end)
            outputs = model(x_batch)
            loss_xent = criterion(outputs, y_batch)
            
            loss = loss_xent 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Learning rate scheduling
        scheduler.step()

        # Logging and checkpointing
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')
            if args.is_binary:
                std,adv,_ = my_eval(data=test_loader, model=model, attack=attack)
                test_acc_track.append(std)
                adv_test_acc_track.append(adv)
                pickle.dump(test_acc_track,open(args.model_folder+'/test_acc_trackft.pkl','wb'))
                pickle.dump(adv_test_acc_track,open(args.model_folder+'/adv_test_acc_trackft.pkl','wb'))
            else: 
                std,adv = my_eval(data=test_loader, model=model, attack=attack)
                test_acc_track.append(std)
                adv_test_acc_track.append(adv)
                pickle.dump(test_acc_track,open(args.model_folder+'/test_acc_trackft.pkl','wb'))
                pickle.dump(adv_test_acc_track,open(args.model_folder+'/adv_test_acc_trackft.pkl','wb'))
        if (epoch+1) % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join('./result/transfer/finetunebase/', f'checkpointforstandardft_{epoch}.pt'))

    # Evaluation
    if args.is_binary:
        std,adv,_ = my_eval(data=test_loader, model=model, attack=attack)
    else: 
        std,adv = my_eval(data=test_loader, model=model, attack=attack)
elif args.Mode == "distill":
    
    print('==> Building model..')
    if args.pretrained != '':
        if args.teacher_model =='res':
            teacher_net = ResNet18(numofclass,args = args,).to(device)
            teacher_net = teacher_net.to(device)
            for param in teacher_net.parameters():
                param.requires_grad = False
        elif args.teacher_model =='wide':
            teacher_net = WideResNet(num_classes=numofclass,depth=args.depth,widen_factor=args.widening,args=args).to(device)
            teacher_net = teacher_net.to(device)
            for param in teacher_net.parameters():
                param.requires_grad = False

    print('==> Loading teacher..')
    teacher_net.load_state_dict(torch.load(args.pretrained))
    teacher_net.eval()

    KL_loss = nn.KLDivLoss()
    XENT_loss = nn.CrossEntropyLoss()
    lr=args.lr

    def train(model,epoch, optimizer):
        net.train()
        train_loss = 0
        train_accuracy = 0
        num = 0
        state = None
        iterator = tqdm(train_loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)  
            data_time.update(time.time() - end)
            if args.datatype == 'network':
                inputs = inputs.unsqueeze(1)
                
                inputs = inputs.unsqueeze(3)   
            optimizer.zero_grad()
            outputs, pert_inputs = net(inputs, targets)
            teacher_outputs = teacher_net(inputs)
            basic_outputs = model(inputs)
            output = torch.max(basic_outputs, dim=1)[1]
            train_accuracy += evaluate(output.cpu().numpy(),targets.cpu().numpy(),"sum")
            loss = args.alpha*args.temp*args.temp*KL_loss(F.log_softmax(outputs/args.temp, dim=1),F.softmax(teacher_outputs/args.temp, dim=1))+(1.0-args.alpha)*XENT_loss(basic_outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num += output.shape[0]
            iterator.set_description(str(loss.item()))
        if (epoch+1)%args.epoch_step == 0:
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
        print('Mean Training Loss:', train_loss/len(iterator))
        print(f"Train accuracy: {train_accuracy/num}")
        return train_loss,train_accuracy/num, state

    def test(epoch, optimizer):
        net.eval()
        adv_correct = 0
        natural_correct = 0
        nc = 0
        total = 0
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
        with torch.no_grad():
            iterator = tqdm(test_loader, ncols=0, leave=False)
            for batch_idx, (inputs, targets) in enumerate(iterator):
                inputs, targets = inputs.to(device), targets.to(device)
                if args.datatype == 'network':
                    inputs = inputs.unsqueeze(1)
                    
                    inputs = inputs.unsqueeze(3)   
                adv_outputs, pert_inputs = net(inputs, targets)
                natural_outputs = model(inputs)
                noise = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=inputs.shape), dtype=torch.float32).to(inputs.device)
                noutput = model(inputs+noise)
                _, adv_predicted = adv_outputs.max(1)
                _, natural_predicted = natural_outputs.max(1)
                _, n = noutput.max(1)
                pred = torch.max(natural_outputs,dim=1)[1]
                advpred = torch.max(adv_outputs,dim=1)[1]
                predn = torch.max(noutput,dim=1)[1]
                totalf1+= f1_score1(targets.cpu().numpy(),pred.cpu().numpy())
                totalf1adv+= f1_score1(targets.cpu().numpy(),advpred.cpu().numpy())
                totalrecall+= recall_score1(targets.cpu().numpy(),pred.cpu().numpy())
                totalrecalladv+= recall_score1(targets.cpu().numpy(),advpred.cpu().numpy())
                totalprecision+= precision_score1(targets.cpu().numpy(),pred.cpu().numpy())
                totalprecisionadv+= precision_score1(targets.cpu().numpy(),advpred.cpu().numpy())
                fp,fn = fpandfn1(pred.cpu().numpy(),targets.cpu().numpy())
                fpr += fp
                fnr += fn
                fp,fn = fpandfn1(advpred.cpu().numpy(),targets.cpu().numpy())
                fpradv += fp
                fnradv += fn
                natural_correct += natural_predicted.eq(targets).sum().item()
                nc += n.eq(targets).sum().item()
                total += targets.size(0)
                adv_correct += adv_predicted.eq(targets).sum().item()
                iterator.set_description(str(adv_predicted.eq(targets).sum().item()/targets.size(0)))
                batchnum +=1
        robust_acc = 100.*adv_correct/total
        natural_acc = 100.*natural_correct/total
        naturalc = 100* nc/total
        print('Natural acc:', natural_acc)
        print('Robust acc:', robust_acc)
        print('Natural Corruption Acc: ', naturalc)
        if args.is_binary:
            all = [totalf1/batchnum,totalf1adv/batchnum,totalprecision/batchnum,totalprecisionadv/batchnum,totalrecall/batchnum,totalrecalladv/batchnum,fpr/batchnum,fpradv/batchnum,fnr/batchnum,fnradv/batchnum]
            print(f"F1:{all[0]}, F1adv:{all[1]}, Precision:{all[2]}, Precisionadv:{all[3]},Recall:{all[4]}, recalladv:{all[5]},FPR:{all[6]}, FPRadv:{all[7]},FNR: {all[8]}, FNRadv{all[9]}")
            return natural_acc, robust_acc,all
        else:
            return natural_acc, robust_acc
    lr = args.lr
    if args.learning_model == "res":
        basic_net = ResNet18(numofclass=numofclass,args = args)
    elif args.learning_model == 'mobilenet':
        basic_net = MobileNet(args = args,num_classes=numofclass) 
    elif args.learning_model == 'wide':
        basic_net = WideResNet(num_classes=numofclass,depth =args.depth,widen_factor=args.widening,args=args)
    elif args.learning_model == 'densenet':
        class Bottleneck(nn.Module):
            def __init__(self, in_planes, growth_rate):
                super(Bottleneck, self).__init__()
                self.bn1 = nn.BatchNorm2d(in_planes)
                self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
                self.bn2 = nn.BatchNorm2d(4*growth_rate)
                self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

            def forward(self, x):
                out = self.conv1(F.relu(self.bn1(x)))
                out = self.conv2(F.relu(self.bn2(out)))
                out = torch.cat([out,x], 1)
                return out
        basic_net = DenseNet(Bottleneck, [6,12,24,16], growth_rate=12,num_classes=numofclass)
    basic_net = basic_net.to(device)
    config = {
        'epsilon': args.eps,
        'num_steps': args.max_iter,
        'step_size': args.eps_step,
    }
    net = AttackPGD(basic_net, config)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    state = None
    std = []
    adv = []
    overall = 0
    for epoch in range(args.epoch):
        lr_current = lr_decay(epoch, args.epoch)
        optimizer.param_groups[0].update(lr=lr_current)
        train_loss1, train_accuracy,state= train(basic_net,epoch, optimizer)
        if (epoch+1)%args.epoch_step == 0:
            if args.is_binary:
                natural_val, robust_val,_ = test(epoch, optimizer)
            else:
                natural_val, robust_val = test(epoch, optimizer)
            std.append(natural_val)
            adv.append(robust_val)
            if epoch == 0:
                overall += natural_val + robust_val
            if (natural_val + robust_val) >= overall:
                torch.save(state, 'result/transfer' +'/distillbest.pth')
        
    with open(args.model_folder +"/standardhistory.pkl","wb") as file:
        pickle.dump(std,file)
    with open(args.model_folder +"/advhistory.pkl","wb") as file:
        pickle.dump(adv,file)
                
    torch.save(state, 'result/transfer'+'/final.pkl')
    
    
elif args.Mode == "kd":
    
    print('==> Building model..')
    if args.pretrained != '':
        if args.teacher_model =='res':
            teacher_net = ResNet18(numofclass,args = args,).to(device)
            teacher_net = teacher_net.to(device)
            for param in teacher_net.parameters():
                param.requires_grad = False
        elif args.teacher_model =='wide':
            teacher_net = WideResNet(num_classes=numofclass,depth=args.depth,widen_factor=args.widening,args=args).to(device)
            teacher_net = teacher_net.to(device)
            for param in teacher_net.parameters():
                param.requires_grad = False

    print('==> Loading teacher..')
    teacher_net.load_state_dict(torch.load(args.pretrained))
    teacher_net.eval()

    KL_loss = nn.KLDivLoss()
    XENT_loss = nn.CrossEntropyLoss()
    lr=args.lr

    def train(model,epoch, optimizer):
        net.train()
        train_loss = 0
        train_accuracy = 0
        num = 0
        state = None
        iterator = tqdm(train_loader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)  
            data_time.update(time.time() - end)
            if args.datatype == 'network':
                inputs = inputs.unsqueeze(1)
                
                inputs = inputs.unsqueeze(3)   
            optimizer.zero_grad()
            teacher_outputs = teacher_net(inputs)
            basic_outputs = model(inputs)
            output = torch.max(basic_outputs, dim=1)[1]
            train_accuracy += evaluate(output.cpu().numpy(),targets.cpu().numpy(),"sum")
            loss = args.alpha*KL_loss(F.log_softmax(basic_outputs/args.temp, dim=1),F.softmax(teacher_outputs/args.temp, dim=1))+(1.0-args.alpha)*XENT_loss(basic_outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num += output.shape[0]
            iterator.set_description(str(loss.item()))
        if (epoch+1)%args.epoch_step == 0:
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
        print('Mean Training Loss:', train_loss/len(iterator))
        print(f"Train accuracy: {train_accuracy/num}")
        return train_loss,train_accuracy/num, state

    def test(model,epoch, optimizer):
        net.eval()
        adv_correct = 0
        natural_correct = 0
        nc = 0
        total = 0
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
        with torch.no_grad():
            iterator = tqdm(test_loader, ncols=0, leave=False)
            for batch_idx, (inputs, targets) in enumerate(iterator):
                inputs, targets = inputs.to(device), targets.to(device)
                if args.datatype == 'network':
                    inputs = inputs.unsqueeze(1)
                    
                    inputs = inputs.unsqueeze(3)   
                adv_outputs,_ = net(inputs, targets)
                natural_outputs = model(inputs)
                noise = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=inputs.shape), dtype=torch.float32).to(inputs.device)
                noutput = model(inputs+noise)
                _, adv_predicted = adv_outputs.max(1)
                _, natural_predicted = natural_outputs.max(1)
                _, n = noutput.max(1)
                pred = torch.max(natural_outputs,dim=1)[1]
                advpred = torch.max(adv_outputs,dim=1)[1]
                predn = torch.max(noutput,dim=1)[1]
                totalf1+= f1_score1(targets.cpu().numpy(),pred.cpu().numpy())
                totalf1adv+= f1_score1(targets.cpu().numpy(),advpred.cpu().numpy())
                totalrecall+= recall_score1(targets.cpu().numpy(),pred.cpu().numpy())
                totalrecalladv+= recall_score1(targets.cpu().numpy(),advpred.cpu().numpy())
                totalprecision+= precision_score1(targets.cpu().numpy(),pred.cpu().numpy())
                totalprecisionadv+= precision_score1(targets.cpu().numpy(),advpred.cpu().numpy())
                fp,fn = fpandfn1(pred.cpu().numpy(),targets.cpu().numpy())
                fpr += fp
                fnr += fn
                fp,fn = fpandfn1(advpred.cpu().numpy(),targets.cpu().numpy())
                fpradv += fp
                fnradv += fn
                natural_correct += natural_predicted.eq(targets).sum().item()
                nc += n.eq(targets).sum().item()
                total += targets.size(0)
                adv_correct += adv_predicted.eq(targets).sum().item()
                iterator.set_description(str(adv_predicted.eq(targets).sum().item()/targets.size(0)))
                batchnum +=1
        robust_acc = 100.*adv_correct/total
        natural_acc = 100.*natural_correct/total
        naturalc = 100* nc/total
        print('Natural acc:', natural_acc)
        print('Robust acc:', robust_acc)
        print('Natural Corruption Acc: ', naturalc)
        if args.is_binary:
            all = [totalf1/batchnum,totalf1adv/batchnum,totalprecision/batchnum,totalprecisionadv/batchnum,totalrecall/batchnum,totalrecalladv/batchnum,fpr/batchnum,fpradv/batchnum,fnr/batchnum,fnradv/batchnum]
            print(f"F1:{all[0]}, F1adv:{all[1]}, Precision:{all[2]}, Precisionadv:{all[3]},Recall:{all[4]}, recalladv:{all[5]},FPR:{all[6]}, FPRadv:{all[7]},FNR: {all[8]}, FNRadv{all[9]}")
            return natural_acc, robust_acc,all
        else:
            return natural_acc, robust_acc
    lr = args.lr
    if args.learning_model == "res":
        basic_net = ResNet18(numofclass=numofclass,args = args)
    elif args.learning_model == 'mobilenet':
        basic_net = MobileNet(args = args,num_classes=numofclass) 
    elif args.learning_model == 'wide':
        basic_net = WideResNet(num_classes=numofclass,depth =args.depth,widen_factor=args.widening,args=args)
    elif args.learning_model == 'densenet':
        class Bottleneck(nn.Module):
            def __init__(self, in_planes, growth_rate):
                super(Bottleneck, self).__init__()
                self.bn1 = nn.BatchNorm2d(in_planes)
                self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
                self.bn2 = nn.BatchNorm2d(4*growth_rate)
                self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

            def forward(self, x):
                out = self.conv1(F.relu(self.bn1(x)))
                out = self.conv2(F.relu(self.bn2(out)))
                out = torch.cat([out,x], 1)
                return out
        basic_net = DenseNet(Bottleneck, [6,12,24,16], growth_rate=12,num_classes=numofclass)
    basic_net = basic_net.to(device)
    
    config = {
        'epsilon': args.eps,
        'num_steps': args.max_iter,
        'step_size': args.eps_step,
    }
    net = AttackPGD(basic_net, config)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    state = None
    std = []
    adv = []
    overall = 0
    for epoch in range(args.epoch):
        lr_current = lr_decay(epoch, args.epoch)
        optimizer.param_groups[0].update(lr=lr_current)
        train_loss1, train_accuracy,state= train(basic_net,epoch, optimizer)
        if (epoch+1)%args.epoch_step == 0:
            if args.is_binary:
                natural_val, robust_val,_ = test(basic_net,epoch, optimizer)
            else:
                natural_val, robust_val = test(basic_net,epoch, optimizer)
            std.append(natural_val)
            adv.append(robust_val)
            if epoch == 0:
                overall += natural_val + robust_val
            if (natural_val + robust_val) >= overall:
                torch.save(state, 'result/transfer' +'/distillbest.pth')
        
    with open(args.model_folder +"/standardhistory.pkl","wb") as file:
        pickle.dump(std,file)
    with open(args.model_folder +"/advhistory.pkl","wb") as file:
        pickle.dump(adv,file)
                
    torch.save(state, 'result/transfer'+'/final.pkl')
elif args.Mode == "distillaad":
    def get_auto_fname(args):
        names = args.method

        if args.method in ['Plain_Clean', 'Plain_Madry']:
            names = names + '-%s' % args.model

        if args.method == 'AdaAD_with_IAD1':
            names = names + '-T(%s)-S(%s)' % (args.teacher_model, args.model) + \
                '-temp(%s)-begin(%s)-alpha(%s)-beta(%s)' % (args.temp,
                                                            args.IAD_begin, args.IAD_alpha, args.IAD_beta)

        if args.method in ['AdaAD']:
            names = names + '-T(%s)-S(%s)' % (args.teacher_model, args.model) + '-alpha(%.4f)' % args.AdaAD_alpha

        if args.method != 'Plain_Clean':
            names = names + '-'.join(['-eps(%d)' % args.eps, 's_eps(%d)' %
                                    args.eps_step, 'n_steps(%d)' % args.max_iter])

        names = names + '-'.join(['-epochs(%s)' % str(args.epoch), 'bs(%s)' % str(args.batch), 'optim(%s)' %
                                args.optim, 'lr(%s)' % str(args.lr), 'lr_sche(%s)' % args.lr_schedule,'pretrained(%s)'% args.pretrained])

        if args.warmup_lr:
            names = names + '-warmup(%s)' % str(args.warmup_lr_epoch)

        if args.is_desc:
            names = names + '-(%s)' % args.desc_str

        return names


    file_name = get_auto_fname(args)

    save_path = os.path.join("./result/transfer/distillaad", file_name)

    print('Save path:%s' % save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Model
    if args.model == 'mobilenet':
        net = MobileNet(args = args,num_classes=numofclass)
    elif args.model == 'res':
        net = ResNet18(args = args,numofclass = numofclass)
    elif args.model == 'wide':
        net = WideResNet(num_classes=numofclass,depth = args.depth,widen_factor= args.wideing,args=args)
    else:
        raise NotImplementedError


    net.to(device)


    if args.dataset == 'cifar10':
        if args.method not in ['Plain_Clean', 'Plain_Madry']:
            if args.teacher_model == 'res':
                teacher_net = model
                teacher_path = args.pretrained
                pretrained_dict = torch.load(teacher_path)
                train_vars_last = teacher_net.linear.parameters()
                model_dict = teacher_net.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
            elif args.teacher_model == 'wide':
                teacher_net = model
                teacher_path = args.pretrained
                pretrained_dict = torch.load(teacher_path)
                train_vars_last = teacher_net.fc.parameters()
                model_dict = teacher_net.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
            
            print('==> Loading teacher..')
            teacher_net.load_state_dict(model_dict)
            teacher_net.to(device)
            teacher_net.eval()

    elif args.dataset == 'cifar100':
        if args.method not in ['Plain_Clean', 'Plain_Madry']:
            
            teacher_net = model
            
            teacher_state_dict = torch.load(args.pretrained)

            
            
            print('==> Loading teacher..')
            teacher_net.load_state_dict(teacher_state_dict)
            teacher_net.to(device)
            teacher_net.eval()

    else:
        if args.method not in ['Plain_Clean', 'Plain_Madry']:
            if args.teacher_model == 'res':
                teacher_net = ResNet18(args = args, numofclass=numofclass).to(device)
                teacher_path = args.pretrained
                pretrained_dict = torch.load(teacher_path)
                train_vars_last = teacher_net.linear.parameters()
                model_dict = teacher_net.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
            elif args.teacher_model == 'wide':
                teacher_net = WideResNet(args,num_classes=numofclass,depth=args.depth,widen_factor=args.widening).to(device)
                teacher_path = args.pretrained
                pretrained_dict = torch.load(teacher_path)
                train_vars_last = teacher_net.fc.parameters()
                model_dict = teacher_net.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
            
            print('==> Loading teacher..')
            teacher_net.load_state_dict(model_dict)
            teacher_net.eval()




    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


    # setup checkpoint
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(
            save_path, 'best_PGD10_acc_model.pth'))

        net.load_state_dict(checkpoint['net'])
        best_Test_acc = checkpoint['clean_acc']
        best_Test_PGD10_acc = checkpoint['PGD10_acc']
        best_Test_acc_epoch = checkpoint['epoch']
        start_epoch = checkpoint['epoch'] + 1

    else:
        start_epoch = 0
        best_Test_acc = 0
        best_Test_NC = 0
        best_Test_Clean_acc_epoch = 0
        best_Test_PGD10_acc = 0
        best_Test_PGD10_acc_epoch = 0

        print('==> Preparing %s %s %s' % (args.model, args.dataset, args.method))
        print('==> Building model..')


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        global Train_acc
        global train_loss

        train_loss = 0
        correct_ori = 0
        total = 0

        net.train()

        lr_current = lr_decay(epoch, args.epoch)
        optimizer.param_groups[0].update(lr=lr_current)
        print('learning_rate: %s' % str(lr_current))

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            
            inputs, targets = inputs.to(device), targets.to(device)
            data_time.update(time.time() - end)
            if args.datatype == 'network':
                inputs = inputs.unsqueeze(1)
                inputs = inputs.unsqueeze(3)
            if args.method == 'Plain_Clean':
                ori_outputs = net(inputs)
                loss = nn.CrossEntropyLoss()(ori_outputs, targets)
                optimizer.zero_grad()

            elif args.method == 'Plain_Madry':
                adv_inputs = Madry_PGD(net, inputs, targets, step_size=args.eps_step,
                                    steps=args.max_iter, epsilon=args.eps)

                net.train()
                optimizer.zero_grad()

                adv_outputs = net(adv_inputs)
                adv_loss = nn.CrossEntropyLoss()(adv_outputs, targets)

                ori_outputs = net(inputs)
                ori_loss = nn.CrossEntropyLoss()(ori_outputs, targets)

                if args.mixture:
                    loss = args.mixture_alpha*ori_loss + \
                        (1-args.mixture_alpha)*adv_loss
                else:
                    loss = adv_loss

            elif args.method == 'AdaAD':
                adv_inputs = adaad_inner_loss(net, teacher_net, inputs, step_size=args.eps_step,
                                            steps=args.max_iter, epsilon=args.eps)

                net.train()
                optimizer.zero_grad()

                ori_outputs = net(inputs)
                adv_outputs = net(adv_inputs)

                with torch.no_grad():
                    teacher_net.eval()
                    t_ori_outputs = teacher_net(inputs)
                    t_adv_outputs = teacher_net(adv_inputs)

                if args.dataset == 'cifar10':
                    kl_loss1 = nn.KLDivLoss()(F.log_softmax(adv_outputs, dim=1),
                                            F.softmax(t_adv_outputs.detach(), dim=1))
                    kl_loss2 = nn.KLDivLoss()(F.log_softmax(ori_outputs, dim=1),
                                            F.softmax(t_ori_outputs.detach(), dim=1))

                elif args.dataset == 'cifar100':
                    kl_loss1 = (1/len(adv_outputs))*torch.sum(nn.KLDivLoss(reduce=False)(
                        F.log_softmax(adv_outputs, dim=1), F.softmax(t_adv_outputs.detach(), dim=1)))
                    kl_loss2 = (1/len(adv_outputs))*torch.sum(nn.KLDivLoss(reduce=False)(
                        F.log_softmax(ori_outputs, dim=1), F.softmax(t_ori_outputs.detach(), dim=1)))
                else:
                    kl_loss1 = nn.KLDivLoss()(F.log_softmax(adv_outputs, dim=1),
                                            F.softmax(t_adv_outputs.detach(), dim=1))
                    kl_loss2 = nn.KLDivLoss()(F.log_softmax(ori_outputs, dim=1),
                                            F.softmax(t_ori_outputs.detach(), dim=1))

                loss = args.AdaAD_alpha*kl_loss1 + (1-args.AdaAD_alpha)*kl_loss2

            elif args.method == 'AdaAD_with_IAD1':
                optimizer.zero_grad()
                adv_inputs = adaad_inner_loss(net, teacher_net, inputs, step_size=args.eps_step,
                                            steps=args.max_iter, epsilon=args.eps)
                net.train()
                ori_outputs = net(inputs)
                adv_outputs = net(adv_inputs)
                Alpha = torch.ones(len(inputs)).cuda()

                # basicop = net(adv_inputs).detach()
                guide = teacher_net(adv_inputs)
                teacher_outputs = teacher_net(adv_inputs)

                KL_loss = nn.KLDivLoss(reduce=False)
                XENT_loss = nn.CrossEntropyLoss()

                if epoch >= args.IAD_begin:
                    for pp in range(len(adv_outputs)):

                        L = F.softmax(guide, dim=1)[pp][targets[pp].item()]
                        L = L.pow(args.IAD_beta).item()
                        Alpha[pp] = L
                    loss = args.IAD_alpha*args.temp*args.temp*(1/len(adv_outputs))*torch.sum(KL_loss(F.log_softmax(adv_outputs/args.temp, dim=1), F.softmax(teacher_outputs/args.temp, dim=1)).sum(dim=1)) + args.IAD_alpha*(
                        1/len(adv_outputs))*torch.sum(KL_loss(F.log_softmax(adv_outputs, dim=1), F.softmax(net(inputs), dim=1)).sum(dim=1).mul(1-Alpha))+(1.0-args.IAD_alpha)*XENT_loss(net(inputs), targets)
                else:
                    loss = args.IAD_alpha*args.temp*args.temp*(1/len(adv_outputs))*torch.sum(KL_loss(F.log_softmax(adv_outputs/args.temp, dim=1), F.softmax(
                        teacher_outputs/args.temp, dim=1)).sum(dim=1))+(1.0-args.IAD_alpha)*XENT_loss(net(inputs), targets)

            else:
                raise NotImplementedError

            loss.backward()
            optimizer.step()

            train_loss += loss.data

            correct_ori += torch.max(ori_outputs, 1)[1].eq(targets.data).cpu().sum()
            total += targets.size(0)

            utils.progress_bar(
                batch_idx,
                len(train_loader),
                'Total_Loss: %.3f| Clean Acc: %.3f%%(%d/%d)'
                '' % (train_loss / (batch_idx + 1),
                    100. * float(correct_ori) / total,
                    correct_ori,
                    total))

        Train_acc = 100. * float(correct_ori) / total


    def test(epoch,std,adv):
        global Test_acc
        global best_Test_acc
        global best_Test_acc_epoch
        global Test_PGD10_acc
        global best_Test_PGD10_acc
        global best_Test_PGD10_acc_epoch
        global test_loss
        global best_Test_NC
        test_loss = 0
        correct_ori = 0
        correct_PGD10 = 0
        correct_total = 0
        correct_n = 0
        total = 0
        totalf1 = 0.0
        totalrecall = 0.0
        totalprecision = 0.0
        fnr = 0.0
        fpr= 0.0
        totalf1adv = 0.0
        totalrecalladv = 0.0
        totalprecisionadv = 0.0
        fnradv = 0.0
        fpradv= 0.0
        batchnum = 0
        net.eval()

        adversary = LinfPGDAttack(net, loss_fn=nn.CrossEntropyLoss(),
                                eps=8/255, nb_iter=10, eps_iter=2/255, rand_init=True, clip_min=0., clip_max=1., targeted = args.target)

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            if args.datatype == 'network':
                inputs = inputs.unsqueeze(1)
                inputs = inputs.unsqueeze(3)
            ori_outputs = net(inputs)
            noise = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=inputs.shape), dtype=torch.float32).to(inputs.device)
            noutput = net(inputs+noise)
            loss = nn.CrossEntropyLoss()(ori_outputs, targets)

            test_loss += loss.data

            correct_ori += torch.max(ori_outputs,
                                    1)[1].eq(targets.data).cpu().sum()
            total += targets.size(0)
            correct_n += torch.max(noutput,
                                    1)[1].eq(targets.data).cpu().sum()
            
            if args.target != False:
                a = adversary.perturb(input[targets == 0], targets,y=1)
                b = adversary.perturb(input[targets != 0], targets,y=0)
                adv_PGD10 = torch.rand(size=input.shape)
                adv_PGD10[targets == 0] = a
                adv_PGD10[targets != 0] = b
            else:
                adv_PGD10 = adversary.perturb(inputs, targets)
            adv_PGD10_outputs = net(adv_PGD10)

            correct_PGD10 += torch.max(adv_PGD10_outputs,
                                    1)[1].eq(targets.data).cpu().sum()
            correct_total = total
            pred = torch.max(ori_outputs,dim=1)[1]
            adv_pred = torch.max(adv_PGD10_outputs,dim = 1)[1]
            totalf1 += f1_score1(targets.cpu().numpy(),pred.cpu().numpy())
            totalprecision+=precision_score1(targets.cpu().numpy(),pred.cpu().numpy())
            totalrecall +=recall_score1(targets.cpu().numpy(),pred.cpu().numpy())
            fp,fn = fpandfn1(pred.cpu().numpy(), targets.cpu().numpy())
            fnr+= fp
            fpr+= fn
            totalf1adv += f1_score1(targets.cpu().numpy(),adv_pred.cpu().numpy())
            totalprecisionadv+=precision_score1(targets.cpu().numpy(),adv_pred.cpu().numpy())
            totalrecalladv +=recall_score1(targets.cpu().numpy(),adv_pred.cpu().numpy())
            fp,fn = fpandfn1(adv_pred.cpu().numpy(), targets.cpu().numpy())
            fnradv+= fp
            fpradv+= fn

            utils.progress_bar(
                batch_idx,
                len(test_loader),
                'Total_Loss: %.3f| Clean Acc: %.3f%%|(%d/%d)| PGD10 Acc: %.3f%%|(%d/%d)| Natural Corruption Acc: %.3f%%|(%d/%d)'
                '' % (test_loss / (batch_idx + 1),
                    100. * float(correct_ori) / total,
                    correct_ori,
                    total,
                    100. * float(correct_PGD10) / correct_total,
                    correct_PGD10,
                    correct_total,
                    100. * float(correct_n) / correct_total,
                    correct_n,
                    correct_total
                    ))
            batchnum +=1
        # Save checkpoint.
        Test_acc = 100. * float(correct_ori) / total
        Test_PGD10_acc = 100. * float(correct_PGD10) / correct_total
        nc = 100. * float(correct_n) / correct_total
        if args.is_binary:
            all = [totalf1/batchnum,totalf1adv/batchnum,totalprecision/batchnum,totalprecisionadv/batchnum,totalrecall/batchnum,totalrecalladv/batchnum,fpr/batchnum,fpradv/batchnum,fnr/batchnum,fnradv/batchnum]
            print(f"F1:{all[0]}, F1adv:{all[1]}, Precision:{all[2]}, Precisionadv:{all[3]},Recall:{all[4]}, recalladv:{all[5]},FPR:{all[6]}, FPRadv:{all[7]},FNR: {all[8]}, FNRadv{all[9]}")
        if Test_acc > best_Test_acc:
            print('Saving..')
            print("best_Test_clean_acc: %0.3f, \tits Test_PGD10_acc: %0.3f" %
                (Test_acc, Test_PGD10_acc))
            state = {
                'net': net.state_dict(),
                'clean_acc': Test_acc,
                'PGD10_acc': Test_PGD10_acc,
                'epoch': epoch,
            }
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(state, os.path.join(save_path, 'best_clean_acc_model.pth'))
            best_Test_acc = Test_acc
            best_Test_acc_epoch = epoch
        if  nc > best_Test_NC:
            best_Test_NC = nc
            
        if Test_PGD10_acc > best_Test_PGD10_acc:
            print('Saving..')
            print("best_Test_PGD10_acc: %0.3f, \tits Test_clean_acc: %0.3f" %
                (Test_PGD10_acc, Test_acc))
            state = {
                'net': net.state_dict(),
                'clean_acc': Test_acc,
                'PGD10_acc': Test_PGD10_acc,
                'epoch': epoch,
            }
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(state, os.path.join(save_path, 'best_PGD10_acc_model.pth'))
            best_Test_PGD10_acc = Test_PGD10_acc
            best_Test_PGD10_acc_epoch = epoch
        if (epoch+1)% args.n_eval_step == 0:
            std.append(Test_acc)
            adv.append(Test_PGD10_acc)
            with open(args.model_folder +"/standardhistory.pkl","wb") as file:
                pickle.dump(std,file)
            with open(args.model_folder +"/advhistory.pkl","wb") as file:
                pickle.dump(adv,file)
        if epoch == args.epoch - 1:
            print('Saving..')
            state = {
                'net': net.state_dict() ,
                'clean_acc': Test_acc,
                'PGD10_acc': Test_PGD10_acc,
                'epoch': epoch,
            }
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(state, os.path.join(
                save_path, 'last_epoch_model(%s).pth' % epoch))


    def main():
        results_log_csv_name =  os.path.join(save_path, 'results.csv')
        # record train log
        with open(results_log_csv_name, 'w') as f:
            f.write(
                'epoch, train_loss, test_loss, train_acc, test_clean_acc, test_PGD10_acc, time\n')

        # start train
        std = []
        adv =[]
        for epoch in range(start_epoch, args.epoch):
            print('current time:', datetime.now().strftime('%b%d-%H:%M:%S'))
            train(epoch)
            test(epoch,std,adv)
            # Log results
            with open(results_log_csv_name, 'a') as f:
                f.write('%5d, %.5f, %.5f, %.5f, %.5f, %.5f, %s,\n'
                        '' % (epoch,
                            train_loss,
                            test_loss,
                            Train_acc,
                            Test_acc,
                            Test_PGD10_acc,
                            datetime.now().strftime('%b%d-%H:%M:%S')))

        print("best_Test_Clean_acc: %.3f" % best_Test_acc)
        print("best_Test_Clean_acc_epoch: %d" % best_Test_acc_epoch)
        print("best_Test_PGD10_acc: %.3f" % best_Test_PGD10_acc)
        print("best_Test_PGD10_acc_epoch: %d" % best_Test_PGD10_acc_epoch)
        print("best_Test_NC_acc: %d" % best_Test_NC)
        # best ACC
        with open(results_log_csv_name, 'a') as f:
            f.write('%s,%03d,%0.3f,%s,%03d,%0.3f,\n' % ('best clean acc (test)',
                                                        best_Test_acc_epoch,
                                                        best_Test_acc,
                                                        'best PGD10 acc (test)',
                                                        best_Test_PGD10_acc_epoch,
                                                        best_Test_PGD10_acc))
    main()
elif args.Mode =="contrastive":
  
    def load_teacher(path,model,args):
        state_dict = torch.load(path)
        
        number = state_dict["fc.weight"].shape[0]
        if model == 'wide':
            teacher_net = WideResNet(num_classes=number,depth=args.depth,widen_factor=args.widening,args=args)
        teacher_net.load_state_dict(state_dict)
        return teacher_net
        
    def main():
        best_acc = 0
        model_t = load_teacher(args.pretrained,args.teacher_model, args)
        model_s = model
        model_t = model_t.cuda()
        opt = torch.optim.SGD(model.parameters(), args.lr, 
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
        criterion = torch.nn.CrossEntropyLoss()

        # Get the shape of a single sample
        single_sample_shape, _ ,_,_= train_dataset[0]
        # Combine to infer overall shape
        overall_shape = (1,single_sample_shape.shape[0],1)
        print(minv,maxv)
        classifier = generate_atk(model,criterion=criterion,optimizer=opt,min=minv,max=maxv,shape= overall_shape,number=numofclass)
        attack = ProjectedGradientDescent(estimator=classifier,norm=args.norm, eps=args.eps,eps_step=args.eps_step,targeted = args.target,max_iter= args.max_iter)
        module_list = nn.ModuleList([])
        module_list.append(model_s)
        trainable_list = nn.ModuleList([])
        trainable_list.append(model_s)
        
        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = nn.KLDivLoss()
        s = single_sample_shape.shape[0]
        print(s)
        if s != args.input_dim:
            data = torch.rand(1, 1, single_sample_shape.shape[0], 1)
            pholder = model_s(data.to(device))
            args.s_dim = model_s.feature.shape[1]
            args.n_data = n_data
            args.t_dim = 40
            criterion_kd = CRDLoss(args,s)
            data = criterion_kd.embed_in_s(data.reshape(1,s))
            pholder = model_t(data.view(1,1,args.input_dim,1).to(device))
            args.t_dim = model_t.feature.shape[1]
        else: 
            
            data = torch.rand(1, 1, single_sample_shape.shape[0], 1)
            pholder = model_s(data.to(device))
            args.s_dim = model_s.feature.shape[1]
            args.n_data = n_data
            pholder = model_t(data.to(device))
            args.t_dim = model_t.feature.shape[1]
            
        criterion_kd = CRDLoss(args,s)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
        if s != args.input_dim:
            module_list.append(criterion_kd.embed_in_s)
            trainable_list.append(criterion_kd.embed_in_s)
        

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)    # classification loss
        criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_kd)     # other knowledge distillation loss
        print(len(trainable_list))
        state_dict = torch.load(args.pretrained)
        number = state_dict["fc.weight"].shape[0]
        # optimizer
        optimizer = optim.SGD(trainable_list.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

        # append teacher after optimizer to avoid weight_decay
        module_list.append(model_t)

        if torch.cuda.is_available():
            module_list.cuda()
            criterion_list.cuda()
            cudnn.benchmark = True

        # routine
        std = []
        adv = []
        dataframetime = pandas.DataFrame({'seed':[],'epochs':[],'time':[]})
        comment=f'Abolition study,gamma_{args.gamma}_cof_{args.cof}_beta_{args.beta}_args.seed_{args.seed},device = {args.device} task = {args.task_name}'
        tb=SummaryWriter(comment=comment)
        for epoch in range(1, args.epoch + 1):
            lr_current = lr_decay(epoch, args.epoch)
            optimizer.param_groups[0].update(lr=lr_current)
            print("==> training...")

            time1 = time.time()
            train_acc, train_loss,df = train_distill(epoch, train_loader, module_list, criterion_list, optimizer, args,dataframetime,s,numofclass,tb)
            time2 = time.time()
            dataframetime = df
            print('epoch {}, total time {:.2f} trainacc{}'.format(epoch, time2 - time1,train_acc))

            

            test_acc, adv_test, test_loss = validate(test_loader, model_s, criterion_cls, numofclass, args,attack,numofclass)
            print(f"Test Accuracy:{test_acc}, Test Loss:{test_loss}, Adv Accuracy: {adv_test} ")
            if (epoch+1)% args.n_eval_step ==0:
                std.append(test_acc)
                adv.append(adv_test)
                with open('Historyforplotting/card'+"/standardhistory.pkl","wb") as file:
                    pickle.dump(std,file)
                with open('Historyforplotting/card' +"/advhistory.pkl","wb") as file:
                    pickle.dump(adv,file)

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                }
                save_file = os.path.join(args.model_folder, '{}_gamma{}_beta{}_cof{}best.pth'.format(args.learning_model,args.gamma,args.beta,args.cof))
                print('saving the best model!')
                torch.save(state, save_file)

            # regular saving
            # if epoch % args.n_eval_step == 0:
            #     print('==> Saving...')
            #     state = {
            #         'epoch': epoch,
            #         'model': model_s.state_dict(),
            #         'accuracy': test_acc,
            #     }
            #     save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}{gamma}{beta}.pth'.format(epoch=epoch,gamma=args.gamma,beta=args.beta))
            #     torch.save(state, save_file)

        # This best accuracy is only for printing purpose.
        # The results reported in the paper/README is from the last epoch. 
        print('best accuracy:', best_acc)
        dataframetime.to_csv(f'result/Contrastive/recordfortime_seed{args.seed}_task_{args.task_name}',index=False)
        # save model
        state = {
            'opt': args,
            'model': model_s.state_dict(),
        }
        save_file = os.path.join(args.model_folder, '{}_gamma{}_beta{}last.pth'.format(args.learning_model,args.gamma,args.beta))
        torch.save(state, save_file)
    # model = load_teacher(args.pretrained,args.teacher_model, args)
    # opt = torch.optim.SGD(model.parameters(), args.lr, 
    #                             weight_decay=args.weight_decay,
    #                             momentum=args.momentum)
    # criterion = torch.nn.CrossEntropyLoss()

    # # Get the shape of a single sample
    # single_sample_shape, _ ,_,_= train_dataset[0]
    # # Combine to infer overall shape
    # overall_shape = (1,single_sample_shape.shape[0],1)
    # print(minv,maxv)
    # classifier = generate_atk(model,criterion=criterion,optimizer=opt,min=minv,max=maxv,shape= overall_shape,number=numofclass)
    # attack = ProjectedGradientDescent(estimator=classifier,norm=args.norm, eps=args.eps,eps_step=args.eps_step,targeted=False,max_iter= args.max_iter)
    # test(model,test_loader,attack)
    main()
# elif args.Mode == 'draw':
#     listforstd = []
#     listforadv = []
#     with open("Historyforplotting/test_acc_tracktwins.pkl",'rb') as file:
#         data = pickle.load(file)
#     with open("Historyforplotting/test_acc_tracktwins.pkl",'rb') as file:
#         data = pickle.load(file)
#     plot_lines(listforstd,"Comparsion Between Methods(Standard)")
#     plot_lines(listforadv,"Comparsion Between Methods(Robust)")
print('Total time: ', data_time.avg)
if not os.path.exists("result/time/timerecord.csv"):
    df = pandas.DataFrame({'seed':[],'method': [],'time':[]})
    df.to_csv("result/time/timerecord.csv",index=False)
else:
    df = pandas.read_csv("result/time/timerecord.csv")
    new_row = pandas.DataFrame({'seed':[args.seed],'method': [args.Mode],'time': [data_time.avg]})
    dataframetime = pandas.concat([df, new_row], ignore_index=True)
    dataframetime.to_csv("result/time/timerecord.csv",index=False)
