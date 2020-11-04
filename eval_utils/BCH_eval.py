import numpy as np
import torch
from time import time
import matplotlib.pyplot as plt
import tifffile

# def BCH_(image):
#     R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    
#     X = np.expand_dims(0.49*R + 0.31*G + 0.2*B, axis=2)/0.17697
#     Y = np.expand_dims(0.17697 * R + 0.8124 *G + 0.01063*B, axis=2)/0.17697
#     Z = np.expand_dims(0.0*R + 0.01 *G + 0.99*B, axis=2)/0.17697
    
#     D = 0.2053*X + 0.7125*Y + 0.467*Z
#     E = 1.9537*X - 1.2797*Y - 0.4429*Z 
#     F = -0.3655*X + 1.012*Y - 0.6104*Z 
    
#     B = np.mean(np.sqrt(D*D + E*E + F*F))
#     return B

def BCH(image):
    r"""
    Brightness Chroma Hue parameter computation function
    Input: torch image with shape of (batch, channel, height, width)
    Output: scalar B
    """
    matrix_XYZ = torch.tensor(((0.49, 0.31, 0.2), 
                               (0.17697, 0.8124, 0.01063), 
                               (0.0, 0.01, 0.99)), dtype = torch.float32)/0.17697
    image = image[0,:,:,:].permute(1,2,0)
    # print(image.shape)
    XYZ = torch.matmul(image, matrix_XYZ)
    
    matrix_DEF = torch.tensor(((0.2053, 0.7125, 0.467), 
                               (1.9537, -1.2797, -0.4429), 
                               (-0.3655, 1.012,-0.6104)), dtype = torch.float32)
    DEF = torch.matmul(XYZ, matrix_DEF)

    B = torch.sqrt(torch.mean(DEF[:,:,0])**2 + 
                   torch.mean(DEF[:,:,1])**2 + 
                   torch.mean(DEF[:,:,2])**2)
    # B = np.mean(np.sqrt(DEF[:,:,0]**2 + DEF[:,:,1]**2 + DEF[:,:,2]**2))
    return B.item()

if __name__ == '__main__':
    img_gt = plt.imread('14_high.png')
    img_in = plt.imread('14_low.png')
    
    img_gt = torch.tensor(img_gt, dtype = torch.float32)
    img_in = torch.tensor(img_in, dtype = torch.float32)
    
    img_gt = torch.unsqueeze(img_gt, dim=0)
    img_in = torch.unsqueeze(img_in, dim=0)

    img_gt = img_gt.permute(0, 3, 1, 2)
    img_in = img_in.permute(0, 3, 1, 2)

    start = time()
    print(BCH(img_gt))
    print("Time: ", time()-start)
    start = time()
    print(BCH(img_in))
    print("Time: ", time()-start)
