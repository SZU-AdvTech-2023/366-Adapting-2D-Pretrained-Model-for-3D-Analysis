import argparse
import os
from PromptModels.pointnet_cls import get_model
from PromptModels.structure import  PViT
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default="0", help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='structure', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=5e-3, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=5e-2, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


args = parse_args()
# 加载您的模型和数据集


checkpoint = torch.load('/home/ubuntu/p3/log/classification/2024-01-10_07-54/checkpoints/best_model.pth')
# # print(checkpoint)
model_state = checkpoint['model_state_dict']
# # print(model_state)
for key,value in model_state.items():
    print(key,value.shape)

new_state_dict = {}
for key in model_state:
    new_key = key
    if key == 'norm.weight':
        new_key = 'enc_norm.weight'
    elif key == 'norm.bias':
        new_key = 'enc_norm.bias'
    new_state_dict[new_key] = model_state[key]

data_path = '/home/ubuntu/p3/data/ModelNet/modelnet40_normal_resampled'
model = PViT(num_classes=40, depth=12, num_heads=12, embed_dim=768).to('cuda')
model.load_state_dict(new_state_dict, True)
test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False,
                                                 num_workers=6)

# 确保模型在评估模式
model.eval()

# 提取特征
features = []
labels = []
with torch.no_grad():
    for data, label in testDataLoader:
        # 如果您的模型是一个CNN，并且您想要提取最后一个全连接层之前的特征
        # 您可能需要调整这里的代码以适应您的模型结构
        data = data.to('cuda')
        label = label.to('cuda')
        output= model(data)  # 假设output是最后一个全连接层之前的特征
        features.extend(output.cpu().numpy())
        labels.extend(label.cpu().numpy())

# 应用t-SNE
features_tsne = TSNE(n_components=2, random_state=0).fit_transform(features)
print('success')
# 绘制t-SNE图
plt.figure(figsize=(12, 8))
scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, alpha=0.5)
plt.xticks([])
plt.yticks([])
plt.savefig('my_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
