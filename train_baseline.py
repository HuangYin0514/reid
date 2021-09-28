
from train_script.baseline_script import train

if __name__ == '__main__':
    print("运行中。。。"+train.__globals__['__name__'])
    print("目标精度94.1%")
    train()
    
    