import os
import torch
from torchvision import transforms, datasets
from PIL import Image
from omegaconf import DictConfig
import hydra
from hydra import utils
from models.FullyConnectedNN import FullyConnectedNN
from models.GoogleNet import GoogleNet
from models.AlexNet5 import AlexNet5
from models.AlexNet8 import AlexNet8
from models.AlexNet12 import AlexNet12
from models.VGG16 import VGG16
from models.VGG19 import VGG19
from models.VGG22 import VGG22
from models.ResNet34 import ResNet34
from models.ResNet101 import ResNet101,ResNet152


@hydra.main(config_path='config', config_name='config', version_base='1.1')
def main(cfg): 
    cwd = utils.get_original_cwd()
    cfg.cwd = cwd  
    # 图片预处理
    transform = transforms.Compose([
        transforms.Resize((cfg.resizex, cfg.resizey)), # AlexNet输入图片大小为224*224 
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载MNIST数据集
    test_data_path = os.path.join(cfg.cwd, 'data')
    mnist = datasets.MNIST(root=test_data_path, train=False, download=True)
    # 加载模型
    __Model__ = {
        'FullyConnectedNN': FullyConnectedNN,
        'AlexNet8': AlexNet8,
        'AlexNet5': AlexNet5,
        'AlexNet12': AlexNet12,
        'VGG16': VGG16 ,
        'VGG19': VGG19,
        'VGG22': VGG22,
        'GoogleNet': GoogleNet, 
        'ResNet': ResNet34,
        'ResNet101': ResNet101,
        'ResNet152': ResNet152
    }
    model = __Model__[cfg.model_name]()
    model.load_state_dict(torch.load(os.path.join(cwd,'save_model',cfg.model_name,'model_epoch_1.pth')))
    model.eval()

    # 取10张图片进行预测
    for i in range(cfg.predict_open,cfg.predict_close):
        img, label = mnist[i]
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)  

        # 使用模型进行预测
        output = model(batch_t)
        _, predicted = torch.max(output, 1)

        print(f'Image {i}: Real class: {label}, Predicted class: {predicted.item()}')

if __name__ == "__main__":
    main()