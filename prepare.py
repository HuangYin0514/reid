import os
from shutil import copyfile
import argparse

######################################################################
# Prepare dataset for training
# You only need to change this line to your dataset download path
# --------------------------------------------------------------------

# opt ==============================================================================
parser = argparse.ArgumentParser(description="Base Dl")
# base (env setting)
parser.add_argument("--download_path", type=str, default="./datasets/Market-1501-v15.09.15")
parser.add_argument("--dataset_save_path", type=str, default="./datasets/Market-1501-v15.09.15")

opt = parser.parse_args()
# util.print_options(opt)


download_path = opt.download_path

if 'cuhk' in download_path:
    suffix = 'png'
else:
    suffix = 'jpg'

if not os.path.isdir(download_path):
    print('please change the download_path')

dataset_save_path = opt.dataset_save_path
if not os.path.isdir(dataset_save_path):
    os.mkdir(dataset_save_path)
    
save_path = dataset_save_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)


# -----------------------------------------
# query
query_path = download_path + '/query'
query_save_path = dataset_save_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:] == suffix:
            continue
        ID = name.split('_')
        src_path = query_path + '/' + name
        dst_path = query_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# -----------------------------------------

# gallery
gallery_path = download_path + '/bounding_box_test'
gallery_save_path = dataset_save_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:] == suffix:
            continue
        ID = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# ---------------------------------------
# train_all
train_path = download_path + '/bounding_box_train'
train_save_path = dataset_save_path + '/pytorch/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == suffix:
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# ---------------------------------------


######################################################################
# change name of the folder(e.g.  0002,0007,0010,0011...  to 0,1,2,3)
# --------------------------------------------------------------------

original_path = save_path

# copy folder tree from source to destination
def copyfolder(src, dst):
    files = os.listdir(src)
    if not os.path.isdir(dst):
        os.mkdir(dst)
    for tt in files:
        copyfile(src + '/' + tt, dst + '/' + tt)


train_save_path = original_path + '/train_all_new'
data_path = original_path + '/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

reid_index = 0
folders = os.listdir(data_path)
for foldernames in folders:
    copyfolder(data_path + '/' + foldernames, train_save_path + '/' + str(reid_index).zfill(4))
    reid_index = reid_index + 1