'''
This file includes the configuration for the experiment. 
You can modify the parameters to run different experiments.
Params:
    model: [GamutMLP, GamutMLP_53KB, GamutMLP_11KB, GamutMLP_137KB]
    pretrained: [True, False]
    pretrained_path: [include path to weights]
    restored_img_folder: [folder name to save restored images]
    device: [cuda, cpu]

Note that the optimizer, learning rate, and iterations are fixed depending on
the model and if it is pretrained or not.
'''
config = {
    'model': 'GamutMLP',
    'pretrained': False,
    'pretrained_path': None,
    # 'restored_img_folder': 'mlp137',
    'device': 'cuda'
}