class initopt():
    img_height =256
    img_width=128
    data_dir='./datasets/Occluded_REID/'
    batch_size=12
    test_batch_size=12
    num_workers=0
opt = initopt


from dataloader.getOccludedReidDataloader import getOccludedreidData

train_loader, query_loader, gallery_loader, num_classes = getOccludedreidData(opt)

print(train_loader)