import numpy as np
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import cv2
import os
import datetime
from canny2image_torch import hackathon

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx]).to("cuda")

def PD(base_img, new_img):
    inception_feature_ref, _ = fid_score.calculate_activation_statistics([base_img], model, batch_size = 1, device="cuda")
    inception_feature, _ = fid_score.calculate_activation_statistics([new_img], model, batch_size = 1, device="cuda")
    pd_value = np.linalg.norm(inception_feature - inception_feature_ref)
    pd_string = F"Perceptual distance to: {pd_value:.2f}"
    print(pd_string)
    return pd_value

scores = []
latencys = []
hk = hackathon()
hk.initialize()
for i in range(20):
    path = "/home/player/pictures_croped/bird_"+ str(i) + ".jpg"
    img = cv2.imread(path)
    start = datetime.datetime.now().timestamp()
    new_img = hk.process(img,
            "a bird", 
            "best quality, extremely detailed", 
            "longbody, lowres, bad anatomy, bad hands, missing fingers", 
            1, 
            256, 
            20,
            False, 
            1, 
            9, 
            2946901, 
            0.0, 
            100, 
            200)
    end = datetime.datetime.now().timestamp()
    print("time cost is: ", (end-start) * 1000)
    now_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(now_dir, "output")
    output_image_dir = os.path.join(now_dir, "output", "image")
    old_img_dir = os.path.join(output_image_dir, "old")
    for dir1 in [output_image_dir, old_img_dir]:
        if not os.path.exists(dir1):
            os.mkdir(dir1)
    new_path = os.path.join(old_img_dir, "bird_"+ str(i) + ".jpg")
    cv2.imwrite(new_path, new_img[0])
    # generate the base_img by running the pytorch fp32 pipeline (origin code in canny2image_TRT.py)
    # base_path = "base_img.jpg"
    #score = PD(base_path, new_path)
    #print("score is: ", score)

