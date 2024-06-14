import pandas as pd
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from hydra import utils
import os


@hydra.main(config_path='config', config_name='config', version_base='1.1')
def main(cfg): 
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd  
    # 从Excel文件中读取数据
    df = pd.read_excel(os.path.join(cwd,'result',cfg.model_name,'result.xlsx'))

    # 获取数据
    epochs = df['Epoch']
    train_losses = df['train_loss']
    test_losses = df['test_loss']
    train_acc_all = df['train_acc']
    test_acc_all = df['test_acc']

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # 绘制训练和测试的损失
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, label='Train Loss')
    ax1.plot(epochs, test_losses, color='tab:orange', label='Test Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper right')

    # 绘制训练和测试的准确率
    color = 'tab:blue'
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, train_acc_all, color=color, label='Train Accuracy')
    ax2.plot(epochs, test_acc_all, color='tab:cyan', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)\
    #右下角图例
    ax2.legend(loc='lower right')

    fig.tight_layout()
    plt.savefig(os.path.join(cwd,'result',cfg.model_name,'loss_and_accuracy.png'))
    plt.show()

if __name__ == "__main__":
    main()
