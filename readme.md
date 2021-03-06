
# BASE-DL

BASE-DL主要构建一个基本的深度学习框架

- [BASE-DL](#base-dl)
  - [训练环境](#训练环境)
  - [实验结果](#实验结果)


## 训练环境  

- 在macbook环境中，命令窗口执行

    ```python 
    cd ~/Documents/project/reid
    conda activate py396
    ```

## 实验结果

|               模型                | kaggle version |  r@1  | occludedreid |                 备注                 |
| :-------------------------------: | :------------: | :---: | :----------: | :----------------------------------: |
|        Resnet_pcb_3branch         |  Version 321   | 93.1  |
|            Resnet_pcb             |  Version 322   | 92.81 |
|         Resnet_pcb_bilstm         |  Version 324   | 93.3  |
|       bilstm_RandomErasing        |  Version 327   | 92.5  |
|               asnet               |  Version 328   |   x   |
|            pcb_256x128            |  Version 336   | 92.4  |
|     pcb_256x128_RandomErasing     |  Version 337   | 92.5  |
|             baseline              |  Version 338   | 83.9  |
|        baseline_exceptData        |  Version 339   | 94.1  |
|          baseline_triks           |  Version 340   | 93.5  |
|      baseline_triks_fixdata       |  Version 342   | 93.8  |
|       baseline_triks_optim        |  Version 344   | 91.9  |
|      baseline_RandomErasing       |  Version 345   | 94.1  |
|    baseline_not—RandomErasing     |  Version 347   | 92.2  |
|       baseline_batchsize64        |  Version 349   | 93.8  |
|           baseline_pcb            |  Version 350   | 93.9  |
|        baseline_pcb-bs128         |  Version 351   | 93.2  |
|   baseline_pcb-normalize_bs128    |  Version 352   | 93.2  |
|      baseline_RandomErasing       |  Version 353   | 94.1  |
|     pcb-256x128_RandomErasing     |  Version 354   | 92.2  |
|  pcb_lstm-256x128_RandomErasing   |  Version 355   | 93.0  |
|        pcb_lstm-ft_lr/100         |  Version 356   | 93.4  |
|            pcb_lstm-ft            |  Version 357   | 92.9  |
|        pcb_lstm-ft_3branch        |  Version 358   | 92.7  |
|    pcb_lstm-ft_3branch_lr/100     |  Version 359   | 93.3  |
| pcb_lstm-ft_3branch_lr/100_weight |  Version 360   | 93.0  |
|  pcb_lstm-ft_3branch_lr/100_fs*0  |  Version 361   | 90.0  |
|      pcb_lstm- occludedreid       |  Version 362   | 92.9  |     52.1     |
|                ffm                |  Version 364   | 85.4  |     58.0     |
|                ffm                |  Version 366   | 84.8  |     63.0     |
|       ffm(bs60) (v278 best)       |  Version 366   | 84.8  |     66.9     |
|            ffm (bs120)            |  Version 368   |       |     64.5     |
|            ffm (bs128)            |  Version 369   |       |     66.4     |
|             baseline              |  Version 371   | 93.9  |              |
|          baseline_apnet           |  Version 379   | 94.2  |              |
|        baseline (ori data)        |  Version 383   | 94.1  |              | 精度上不去的原因是因为数据 transform |


