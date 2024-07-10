import torch

def update_tensor(tensor):
    import time 
    from math import sin
    tensor += sin(time.time())/1000
    return tensor

    
tensor = torch.tensor([[0.1, 0.2, 0.3 ],
                       [0.4, 0.5, 0.6],
                       [0.7, 0.8, 0.9],
                       [1.0, 0.9, 0.8],],
                      dtype=torch.float16,
                      device=torch.device('cuda:0'))

from Float_Advance.Visualize_Float_Tensor_GL_IMGUI import GUI
GUI(tensor, update_tensor).renderOpenGL()