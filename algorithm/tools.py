import torch
from art.estimators.classification import PyTorchClassifier
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
def fpandfn1(pred,true):
    pred = np.array(pred, dtype=bool)
    true = np.array(true, dtype=bool)

    # Calculate confusion matrix elements
    TP = np.sum((pred == 1) & (true == 1))
    FP = np.sum((pred == 1) & (true == 0))
    TN = np.sum((pred == 0) & (true == 0))
    FN = np.sum((pred == 0) & (true == 1))
    fpr = FP / float(FP + TN) if (FP + TN) > 0 else 0
    fnr = FN / float(TP + FN) if (TP + FN) > 0 else 0
    return fpr,fnr
class AttackPGD(nn.Module):
        def __init__(self, basic_net, config):
            super(AttackPGD, self).__init__()
            self.basic_net = basic_net
            self.step_size = config['step_size']
            self.epsilon = config['epsilon']
            self.max_iter = config['num_steps']

        def forward(self, inputs, targets):
            x = inputs.detach()
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            for i in range(self.max_iter):
                x.requires_grad_()
                with torch.enable_grad():
                    loss = F.cross_entropy(self.basic_net(x), targets, size_average=False)
                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + self.step_size*torch.sign(grad.detach())
                x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
                x = torch.clamp(x, 0.0, 1.0)
            return self.basic_net(x), x
def precision_score1(y_true, y_pred):
    y_pred = np.array(y_pred, dtype=bool)
    y_true = np.array(y_true, dtype=bool)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def recall_score1(y_true, y_pred):
    y_pred = np.array(y_pred, dtype=bool)
    y_true = np.array(y_true, dtype=bool)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def f1_score1(y_true, y_pred):
    y_pred = np.array(y_pred, dtype=bool)
    y_true = np.array(y_true, dtype=bool)
    precision_value = precision_score1(y_true, y_pred)
    recall_value = recall_score1(y_true, y_pred)
    
    return 2 * (precision_value * recall_value) / (precision_value + recall_value) if (precision_value + recall_value) != 0 else 0
     
def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()
def generate_atk(model,criterion,optimizer,min:int,max:int,shape,number):
    classifier = PyTorchClassifier(model=model,clip_values=(min, max),loss=criterion,optimizer=optimizer,input_shape= shape,nb_classes=number)
    return classifier
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res 
def plot_lines(list_of_lists,title):
    """
    Plots multiple lines on a graph, where each line is represented by a list of 5 y-values.
    The x-values are assumed to be the indices of each element in the lists (0-4).

    :param list_of_lists: A list containing lists of y-values for each line.
    """
    
    # Generating x-values assuming they are 0-4 (indices of the elements)
    x_values = [10,20,30,40,50]
    
    # Plotting each list as a separate line
    for i, y_values in enumerate(list_of_lists):
        plt.plot(x_values, y_values, label=f'Line {i+1}')
    
    # Adding legend, labels, and title for clarity
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    
    # Displaying the plot
    plt.show()
def get_onlymalicious_set(dataset_x,dataset_y):
    
    condition = dataset_y.astype(bool)

    malicious_dataset_y = np.extract(condition,dataset_y)
    print("malicious_dataset_y.shape:",malicious_dataset_y.shape)
    print("malicious_dataset_y:",malicious_dataset_y)

    cond=np.expand_dims(condition,1)
    cond_expend = np.full((dataset_x.shape[0], dataset_x.shape[1]), False, dtype=bool)
    cond = np.logical_or(cond_expend, cond)        
    
    malicious_dataset_x = np.extract(cond,dataset_x)
    malicious_dataset_x = np.reshape(malicious_dataset_x, (malicious_dataset_y.shape[0], dataset_x.shape[1]))        
    print("malicious_dataset_x.shape:",malicious_dataset_x.shape)
    """ 
    malicious_dataset_x.shape: (43906, 30)
    """
    print("malicious_dataset_y.shape:",malicious_dataset_y.shape)
    
    return malicious_dataset_x, malicious_dataset_y
def get_onlymalicious_set(dataset_x,dataset_y):
    
    # extract malicious set
    condition = dataset_y.astype(bool)
    # print("condition.shape:",condition.shape)
    # print("condition[:10]:",condition[:10])
    """ 
    condition.shape: (94479,)
    """
    
    malicious_dataset_y = np.extract(condition,dataset_y)
    print("malicious_dataset_y.shape:",malicious_dataset_y.shape)
    print("malicious_dataset_y:",malicious_dataset_y)

    # benign_dataset_y = np.extract(1-condition, dataset_y)
    # print("benign_dataset_y.shape:",benign_dataset_y.shape)
    # print("benign_dataset_y:",benign_dataset_y)                                   
    # """ 
    # malicious_dataset_y.shape: (43906,)
    # malicious_dataset_y: [1 1 1 ... 1 1 1]
    # benign_dataset_y.shape: (50573,)
    # benign_dataset_y: [0 0 0 ... 0 0 0]
    # """
    
    cond=np.expand_dims(condition,1)
    # print("cond.shape:",cond.shape)
    # 创建形状为(4233, 41)的全False数组
    cond_expend = np.full((dataset_x.shape[0], dataset_x.shape[1]), False, dtype=bool)
    
    # 将条件数组广播到result数组中
    cond = np.logical_or(cond_expend, cond)        
    # print("cond.shape:",cond.shape)        
    """
    cond.shape: (4233, 1)
    cond.shape: (4233, 41)
    """
    
    malicious_dataset_x = np.extract(cond,dataset_x)
    malicious_dataset_x = np.reshape(malicious_dataset_x, (malicious_dataset_y.shape[0], dataset_x.shape[1]))        
    print("malicious_dataset_x.shape:",malicious_dataset_x.shape)
    """ 
    malicious_dataset_x.shape: (43906, 30)
    """
    print("malicious_dataset_y.shape:",malicious_dataset_y.shape)
    # malicious_dataset_y.shape: (43906,)

    
    
    
    
    return malicious_dataset_x, malicious_dataset_y