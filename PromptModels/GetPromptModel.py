"""
build_promptmodel script  ver: Mar 25th 19:20

"""
import timm
import torch
from setuptools import config

from PromptModels.structure import *


'''
 timm MODEL_ZOO:
 
 'swin_base_patch4_window7_224',
 'swin_base_patch4_window7_224_in22k',
 'swin_base_patch4_window12_384',
 'swin_base_patch4_window12_384_in22k',
 'swin_large_patch4_window7_224',
 'swin_large_patch4_window7_224_in22k',
 'swin_large_patch4_window12_384',
 'swin_large_patch4_window12_384_in22k',
 'swin_small_patch4_window7_224',
 'swin_tiny_patch4_window7_224',

 'visformer_small',
 'vit_base_patch16_224',
 'vit_base_patch16_224_in21k',
 'vit_base_patch16_224_miil',
 'vit_base_patch16_224_miil_in21k',
 'vit_base_patch16_384',
 'vit_base_patch32_224',
 'vit_base_patch32_224_in21k',
 'vit_base_patch32_384',
 'vit_base_r50_s16_224_in21k',
 'vit_base_r50_s16_384',
 'vit_huge_patch14_224_in21k',
 'vit_large_patch16_224',
 'vit_large_patch16_224_in21k',
 'vit_large_patch16_384',
 'vit_large_patch32_224_in21k',
 'vit_large_patch32_384',
 'vit_large_r50_s32_224',
 'vit_large_r50_s32_224_in21k',
 'vit_large_r50_s32_384',
 'vit_small_patch16_224',
 'vit_small_patch16_224_in21k',
 'vit_small_patch16_384',
 'vit_small_patch32_224',
 'vit_small_patch32_224_in21k',
 'vit_small_patch32_384',
 'vit_small_r26_s32_224',
 'vit_small_r26_s32_224_in21k',
 'vit_small_r26_s32_384',
 'vit_tiny_patch16_224',
 'vit_tiny_patch16_224_in21k',
 'vit_tiny_patch16_384',
 'vit_tiny_r_s16_p8_224',
 'vit_tiny_r_s16_p8_224_in21k',
 'vit_tiny_r_s16_p8_384',
'''


def build_promptmodel(point_nums = 1024,num_classes=40, img_size=224, model_idx='ViT', patch_size=16, base_model='vit_base_patch16_224_in21k',
                      Prompt_Token_num=20, VPT_type="Deep",device = 1 ):
    # VPT_type = "Deep" / "Shallow"

    if model_idx[0:3] == 'ViT':
        # ViT_Prompt
        import timm

        basic_model = timm.create_model(base_model,
                                        pretrained=True)
        #字典？
        base_state_dict = basic_model.state_dict()

        #删除头的权重
        del base_state_dict['head.weight']
        #删除头的偏置
        del base_state_dict['head.bias']
        #创建model
        model = VPT_ViT(config,point_nums = point_nums, num_classes=num_classes, img_size=img_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                        VPT_type=VPT_type,device = device)

        model.load_state_dict(base_state_dict, False)
        #model.New_CLS_head(num_classes)
        model.Freeze()
    else:
        print("The model is not difined in the Prompt script")
        return -1

    return model

if __name__ == '__main__':
   print( build_promptmodel() )