import torch
from torch import nn
import torch.nn.functional as F

from PromptModels.pointmlp_part import pointMLP
from PromptModels.utils import Block

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(ConvBNReLU1D, self).__init__()
        self.act = nn.ReLU()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class PViT_seg(nn.Module):

    def __init__(self,num_classes=40, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,act_layer=nn.GELU, npoint=196, radius=0.15, nsample=64,normal=False):
        # Recreate ViT
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.embed = embed_dim
        self.norm = norm_layer(embed_dim)
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.normal = normal

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        self.num_classes = num_classes

        self.encoder = pointMLP()

        self.classifier = nn.Sequential(
            nn.Conv1d(256 + 768, 256, 1, bias=True),
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.Conv1d(256, num_classes, 1, bias=True)
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.embed)
        )
        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3,
                                                        mlp=[self.trans_dim * 4, 1024])

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

        for name, param in self.named_parameters():
            if 'encoder' in name or 'adaptmlp' in name or 'classifier' in name :
                param.requires_grad = True

    def forward_features(self, x, cls_label):
        pass
        # B,N,_ = x.shape
        # # B,256,1568, B,196,768
        # x_encoder,feat= self.encoder(x.permute(0,2,1))
        # # B,2048,768
        for i in range(len(self.blocks)):
            feat = self.blocks[i](feat,temp=1)
        # # B,1,768
        # feat = feat.max(1,keepDim=True)[0] + feat.mean(1,keepDim=True)
        # feat = feat.permute(0,2,1).repeat(1,1,N)
        # feat = torch.cat([])
        # return

    def forward(self, x, cls_label):
        # CLS_LABEL b,1,16
        B, N, _ = x.shape
        norm = x[:,:,3:].permute(0,2,1)
        xyz = x[:,:,:3].permute(0,2,1)
        # B,256,1568, B,196,768
        x_encoder, feat = self.encoder(xyz,norm,cls_label)
        # B,2048,768

        # B,1,768
        feat = feat.max(1, keepdim=True)[0] + feat.mean(1, keepdim=True)
        feat = feat.permute(0, 2, 1).repeat(1, 1, N)
        feat = torch.cat([x_encoder,feat],dim = 1)
        feat = self.classifier(feat)
        return feat.permute(0,2,1)


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss

def build_structure_pointmlp(num_classes=50, model_idx='ViT', base_model='vit_base_patch16_224_in21k',normal = False):

    if model_idx[0:3] == 'ViT':
        import timm
        basic_model = timm.create_model(base_model, pretrained=True)
        base_state_dict = basic_model.state_dict()
        del base_state_dict['head.weight']
        del base_state_dict['head.bias']

        model = PViT_seg(num_classes=num_classes,depth=12, num_heads=12, embed_dim=768)
        model.load_state_dict(base_state_dict, False)
        model.freeze()

    else:
        print("The model is not defined in the Prompt script")
        return -1

    return model

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()

    return new_y

if __name__ == '__main__':

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    num_classes = 16
    num_part = 50
    B = 2
    points = torch.randn(2, 2048, 7).to('cuda')
    cls_label = torch.rand([2, 16]).to('cuda')
    print(cls_label.shape)
    # print(label.shape)
    model = build_structure_pointmlp(num_classes=num_part).to('cuda')
    for name, param in model.named_parameters():
        print(name)
    print("======= hot ========")
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            print(f"{name} : {param.shape}")
    seg_pred = model(points,cls_label)
    # print(to_categorical(label,num_classes).shape)
    print(seg_pred.shape)
