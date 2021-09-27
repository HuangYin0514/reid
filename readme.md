
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

|            模型            | kaggle version |  r@1  |
| :------------------------: | :------------: | :---: |
|     Resnet_pcb_3branch     |  Version 321   | 93.1  |
|         Resnet_pcb         |  Version 322   | 92.81 |
|     Resnet_pcb_bilstm      |  Version 324   | 93.3  |
|    bilstm_RandomErasing    |  Version 327   | 92.5  |
|           asnet            |  Version 328   |   x   |
|        pcb_256x128         |  Version 336   | 92.4  |
| pcb_256x128_RandomErasing  |  Version 337   | 92.5  |
|          baseline          |  Version 338   | 83.9  |
|    baseline_exceptData     |  Version 339   | 94.1  |
|       baseline_triks       |  Version 340   | 93.5  |
|   baseline_triks_fixdata   |  Version 342   | 93.8  |
|    baseline_triks_optim    |  Version 344   | 91.9  |
|   baseline_RandomErasing   |  Version 345   | 94.1  |
| baseline_not—RandomErasing |  Version 347   | 92.2  |
|    baseline_batchsize64    |  Version 349   | 93.8  |
|        baseline_pcb        |  Version 350   | 93.9  |
|     baseline_pcb-bs128     |  Version 350   |       |
