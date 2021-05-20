import os
import cv2
import numpy as np

def visualization(vis_root, img_root, image, target, output):
    vis_root = vis_root
    
    img_path = vis_root + '/' + img_root.split('/')[-1] 
    # write images
    img_copy      = image.clone().permute(1,2,0)
    img = np.uint8(img_copy.cpu())
    gt  = np.uint8(target.permute(1,2,0).cpu())
    output= np.uint8(output.permute(1,2,0).detach().cpu())
    gt  = cv2.applyColorMap(gt, cv2.COLORMAP_JET)
    output= cv2.applyColorMap(output, cv2.COLORMAP_JET)
    
    print(img.shape,gt.shape,output.shape)
    vis = cv2.hconcat([img,gt,output])
    vis = cv2.hconcat([gt,output])
    cv2.imwrite(img_path,vis)