

# 运行命令

- [运行命令](#运行命令)
  - [train](#train)
  - [train_pcb_2branch](#train_pcb_2branch)


## [train](../train.py)

局部分支

```python
!python train.py \
--data_dir=/kaggle/input/market1501/Market-1501-v15.09.15 \
--batch_size=128 \
--test_batch_size=128 \
--num_workers=4 \
--num_epochs=60 \
--img_height=384 \
--img_width=128 
```

[运行结果](https://www.kaggle.com/huangyin123/reid-custom?scriptVersionId=74439901) Testing: top1:0.9231 top5:0.9706 top10:0.9795 mAP:0.7733


## [train_pcb_2branch](../train_2branch.py)

局部分支+全局分支

```python
!python train.py \
--data_dir=/kaggle/input/market1501/Market-1501-v15.09.15 \
--batch_size=128 \
--test_batch_size=128 \
--num_workers=4 \
--num_epochs=60 \
--img_height=384 \
--img_width=128 
```
