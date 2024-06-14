import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import pandas as pd
import hydra
from omegaconf import DictConfig
from hydra import utils
from models.FullyConnectedNN import FullyConnectedNN
from models.AlexNet8 import AlexNet8
from models.AlexNet5 import AlexNet5
from models.AlexNet12 import AlexNet12
from models.VGG16 import VGG16
from models.VGG19 import VGG19
from models.VGG22 import VGG22
from models.ResNet34 import ResNet34
from models.ResNet101 import ResNet101,ResNet152



@hydra.main(config_path='config', config_name='config', version_base='1.1')
def main(cfg: DictConfig):
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd  
    __Model__ = {
        'FullyConnectedNN': FullyConnectedNN,
        'AlexNet8': AlexNet8,
        'AlexNet5': AlexNet5,
        'AlexNet12': AlexNet12,
        'VGG16': VGG16 ,
        'VGG19': VGG19,
        'VGG22': VGG22,
        'ResNet34': ResNet34,
        'ResNet101': ResNet101,
        'ResNet152': ResNet152
    }
    model = __Model__[cfg.model_name]()
    # 1.设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2.数据预处理
    transform = transforms.Compose([
        transforms.Resize((cfg.resizex, cfg.resizey)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data_path = os.path.join(cfg.cwd, 'data')
    test_data_path = os.path.join(cfg.cwd, 'data')
    # 3.加载 MNIST 数据集
    train_dataset = datasets.MNIST(root=train_data_path, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=test_data_path, train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False)
    # 4.初始化模型
    model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # 创建保存模型的文件夹
    save_model = os.path.join(cwd, 'save_model', cfg.model_name)
    if not os.path.exists(save_model):
        os.makedirs(save_model)

#6. 训练和测试模型
    num_epochs = cfg.epochs
    train_losses = []
    test_losses = []
    train_acc_all = []
    test_acc_all = []
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            train_correct += (predicted == target).sum().item()
        
        train_loss = round(train_loss / len(train_loader.dataset),5)
        train_losses.append(train_loss)
        train_acc =  round(train_correct / len(train_loader.dataset),5)
        train_acc_all.append(train_acc)

        # 保存模型
        torch.save(model.state_dict(), os.path.join(save_model, f"model_epoch_{epoch+1}.pth"))
        # 测试
        model.eval()
        test_loss = 0.0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                test_correct += (predicted == target).sum().item()
        test_loss = round(test_loss / len(test_loader.dataset),5)
        test_acc = round(test_correct / len(test_loader.dataset),5)
        test_losses.append(test_loss)
        test_acc_all.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}," 
            f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
         # 保存最终的训练和测试准确率、损失到excel文件
    data = {'train_loss':train_losses,'test_loss':test_losses
            ,'train_acc':train_acc_all,'test_acc':test_acc_all} 
    df = pd.DataFrame(data)
    df.index = range(1,len(df)+1)
    df.index.name = 'Epoch'
    # 保存到excel文件到result文件夹
    result_path = os.path.join(cwd,'result',cfg.model_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    df.to_excel( os.path.join(result_path,'result.xlsx'),index=True)
    print('训练和测试准确率、损失已保存')


  
if __name__ == '__main__':
    main()
   


   

