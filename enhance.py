## Image Enhancer
# Author : Arun Aniyan
# 13th February 2021


import argparse
import os.path
from os import listdir
from libsvm import svmutil  # Trick to fix brisque import error in OSX
from brisque import BRISQUE

import torch

from utils import utils_image as util
from utils import utils_model

from models.network_dncnn import DnCNN as net

## Initiliaze Params ##

# Inference Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Brisq Object
brisq = BRISQUE()

# Model zoo
model_pool = 'model_zoo'

def main(args):

    # Model name
    if args['model'] is None:
        model_name = 'dncnn_50'  # 'dncnn_25' | 'dncnn_50' | 'dncnn_gray_blind' | 'dncnn_color_blind' | 'dncnn3'
    else:
        model_name = args['model']

    # RGB or Gray Scale mode
    if args['color'] == 'rgb':
        model_name = 'dncnn_color_blind'

    # Error Handling for unavailable model
    try:
        model_path = os.path.join(model_pool, model_name + '.pth')
        print("Using model %s" % (model_name))
        print("---------------------------------")

    except:
        print('Model not found')
        exit()

    # Disabled now - Found to reduce quality of output
    x8 = False  # default: False, x8 to boost performance

    if args['type'] == None:
        task_current = 'dn'  # 'dn' for denoising | 'sr' for super-resolution
    else:
        task_current = args['type']

    sf = 1  # unused for denoising

    # Identify if model used is color
    if 'color' in model_name:
        n_channels = 3  # fixed, 1 for grayscale image, 3 for color image
    else:
        n_channels = 1  # fixed for grayscale image

    if model_name in ['dncnn_gray_blind', 'dncnn_color_blind', 'dncnn3']:
        nb = 20  # fixed
    else:
        nb = 17  # fixed


    border = sf if task_current == 'sr' else 0  # shave boader to calculate PSNR and SSIM

    need_H = False

    # Load Model
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # Load Image
    if (args['input'] != None) and (args['batch']==None): # Single image
        img = args['input']
        predict(img,n_channels,model,x8)
    elif (args['input'] == None) and (args['batch'] != None): # Batch Mode
        # Load each image from directory and predict
        mypath = args['batch']
        # Check if path exist and load list of files
        if os.path.exists(mypath):
            files = [f for f in listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
            print("Found %d files" %(len(files)))
            # Predict for each file and save results
            for item in files:
                try:
                    filepath = os.path.join(mypath,item)
                    predict(filepath,n_channels,model,x8)
                except:
                    print("Error with %s"%(item))
        else:
            print("Path does not exist")
            exit()

# Function to load image and return tensor with some extra params
def load_image(infile,n_channels):
    img_name, ext = os.path.splitext(os.path.basename(infile))
    print("Input File: %s"%(img_name+ext))
    img_L = util.imread_uint(infile, n_channels=n_channels)
    img_L = util.uint2single(img_L)
    img_L = util.single2tensor4(img_L)
    print('Brisque Score of input image : %f' % (brisq.get_score(infile)))
    return img_name,ext,img_L

# Load image and do prediction
def predict(img,n_channels,model,x8):
    img_name, ext, img_L = load_image(img, n_channels)
    img_L = img_L.to(device)

    # Prediction
    if not x8:
        img_E = model(img_L)
    else:
        img_E = utils_model.test_mode(model, img_L, mode=3)

    img_E = util.tensor2uint(img_E)

    # Save Image
    out_path = 'testresults'
    util.imsave(img_E, os.path.join(out_path, img_name + ext))
    print('Brisque Score of enhanced image : %f' % (brisq.get_score(os.path.join(out_path,
                                                                                 img_name + ext))))
    print('*-----------------------------------------------*')


# Run Main
if __name__ == '__main__':
    # Set argparser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=False, help="Path to image file")
    ap.add_argument("-b", "--batch", required=False, help="Batch mode <folder path>")
    ap.add_argument("-m", "--model", required=False, help="Model name < dncnn_25 | dncnn_50 "
                                                          "| dncnn_gray_blind | dncnn_color_blind | dncnn3 >")
    ap.add_argument("-c", "--color", required=False, help="rgb or gs")
    ap.add_argument("-t", "--type", required=False, help='Denoise (dn) or Super Resolution (sn)')

    args = vars(ap.parse_args())

    main(args)
