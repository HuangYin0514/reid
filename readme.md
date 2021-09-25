
# BASE-DL

BASE-DL主要构建一个基本的深度学习框架

- [BASE-DL](#base-dl)
  - [训练环境](#训练环境)
  - [实验结果](#实验结果)


## 训练环境  

- 在macbook环境中，命令窗口执行

    ```python 
    cd ~/Documents/project/base-dl
    conda activate py396
    ```

## 实验结果

|         模型         | kaggle version |  r@1  |
| :------------------: | :------------: | :---: |
|  Resnet_pcb_3branch  |  Version 321   | 93.1  |
|      Resnet_pcb      |  Version 322   | 92.81 |
|  Resnet_pcb_bilstm   |  Version 324   | 93.3  |
| bilstm_RandomErasing |  Version 327   | 92.5  |
|        asnet         |  Version 328   |   x   |



    
