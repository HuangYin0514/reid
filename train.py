import argparse

import matplotlib
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

matplotlib.use("agg")
import os
import time

# from PIL import Image
import matplotlib.pyplot as plt

from models import *

from loss import CrossEntropyLabelSmoothLoss,TripletLoss

version = torch.__version__

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
# base (env setting)
parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
parser.add_argument("--name", type=str, default="person_reid")
# data
parser.add_argument(
    "--data_dir", type=str, default="datasets/Market-1501-v15.09.15/pytorch"
)
# parser.add_argument(
#     "--data_dir", type=str, default="./datasets/Market-1501-v15.09.15"
# )
parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--test_batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=0, type=int)
# train
parser.add_argument("--num_epochs", type=int, default=2)
# other
parser.add_argument("--img_height", type=int, default=4)
parser.add_argument("--img_width", type=int, default=2)

opt = parser.parse_args()
# util.print_options(opt)


train_all = '_all'
data_dir = opt.data_dir
img_height= opt.img_height
img_width=opt.img_width
# speed up compution
torch.backends.cudnn.benchmark = True
# device
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("using cuda ...")
name = "reid"
# data ===============================================
transform_train_list = [
    transforms.Resize((img_height, img_width), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

transform_val_list = [
    transforms.Resize(
        size=(img_height, img_width), interpolation=3
    ),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]

data_transforms = {
    "train": transforms.Compose(transform_train_list),
    "val": transforms.Compose(transform_val_list),
}


image_datasets = {}
image_datasets["train"] = datasets.ImageFolder(
    os.path.join(data_dir, "train" + train_all), data_transforms["train"]
)
# image_datasets["val"] = datasets.ImageFolder(
#     os.path.join(data_dir, "val"), data_transforms["val"]
# )


dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )  # 8 workers may work faster
    for x in ["train"]
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes


y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train']/opt.batch_size) # first 5 epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train']:
            # model.train(True)  # Set model to training mode
           
            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = inputs.to(device), labels.to(device)
                now_batch_size,c,h,w = inputs.shape

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)

                sm = nn.Softmax(dim=1)
                part = {}
                num_part = 6
                for i in range(num_part):
                    part[i] = outputs[i]

                score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                _, preds = torch.max(score.data, 1)

                loss = criterion(part[0], labels)

                for i in range(num_part-1):
                    loss += criterion(part[i+1], labels)
                
                del inputs

                loss.backward()
                optimizer.step()

                 # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                del loss
                running_corrects += float(torch.sum(preds == labels.data))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                
                y_loss[phase].append(epoch_loss)
                y_err[phase].append(1.0-epoch_acc)   
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        #print('Best val Acc: {:4f}'.format(best_acc))

        # # load best model weights
        # model.load_state_dict(last_model_wts)
        # save_network(model, 'last')
        return model

######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda()



# model ============================================================================================================
model = Resnet_pcb_3branch(len(class_names))
model = model.to(device)



# criterion ============================================================================================================
criterion = F.cross_entropy
ce_labelsmooth_loss = CrossEntropyLabelSmoothLoss(num_classes=len(class_names))
triplet_loss = TripletLoss(margin=0.3)

# optimizer ============================================================================================================
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lr = 0.1
base_param_ids = set(map(id, model.backbone.parameters()))
new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
param_groups = [
    {"params": model.backbone.parameters(), "lr": lr / 10},
    {"params": new_params, "lr": lr},
]
optimizer = torch.optim.SGD(
    param_groups, momentum=0.9, weight_decay=5e-4, nesterov=True
)

# # scheduler ============================================================================================================
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# if __name__ == '__main__':
#     train_model(None,None,None,None)
