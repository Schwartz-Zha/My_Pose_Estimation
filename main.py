from src.model import Simplified_Pose_Model
from utils.util import *
import torch
import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import time


def Net_Prediction(model, image, device, backbone='SimpleNet'):
    scale_search = [1]
    stride = 8
    padValue = 128
    heatmap_avg = np.zeros((image.shape[0], image.shape[1], 19))

    for m in range(len(scale_search)):
        scale = scale_search[m]
        imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
        # pad right and down corner to make sure image size is divisible by 8
        im = np.transpose(np.float32(imageToTest_padded), (2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)
        data = torch.from_numpy(im).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _heatmap = model(data).cpu()

        # extract outputs, resize, and remove padding
        heatmap = np.transpose(np.squeeze(_heatmap), (1, 2, 0))  # output 1 is heatmaps
        heatmap = cv2.resize(np.float32(heatmap), (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg += heatmap / len(scale_search)


    return heatmap_avg


def draw_pose(canvas, key_points, scale):
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(18):
        y = key_points[i][0] / scale
        x = key_points[i][1] / scale
        #        print("x = " + str(x) + "\n")
        #        print("y = " + str(y) + "\n")
        cv2.circle(canvas, (int(x), int(y)), 10, colors[i], thickness=-1)

    return canvas







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Open Pose Demo')
    parser.add_argument("-image", help='image path', default='images/ski.jpg', type=str)
    parser.add_argument("-scale", help='scale to image', default=0.3, type=float)
    parser.add_argument("-thre", help="threshold for heatmap part", default=0.1, type=str)

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Simplified_Pose_Model()

    pretrained_state_dict = torch.load(os.path.join('weights', 'bodypose_model'))
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)

    model = model.to(device)
    print('model is successfully loaded...')

    model.eval()

    test_image = args.image
    image = cv2.imread(test_image)
    imageToTest = cv2.resize(image, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)

    since = time.time()

    heatmap = Net_Prediction(model, imageToTest, device)
    t1 = time.time()
    print("model inference in {:2.3f} seconds".format(t1 - since))

    print("heatmap size: " + str(heatmap.shape) + "\n")
    key_points = []
    for i in range(18):
        heatmapi = heatmap[:, :, i]
        key_points.append( np.unravel_index(heatmapi.argmax(), heatmapi.shape))

    t2 = time.time()
    print("find peaks in {:2.3f} seconds".format(t2 - t1))


    canvas = draw_pose(image, key_points, args.scale)

    print("total inference in {:2.3f} seconds".format(time.time() - since))

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    plt.savefig('results/res1.png', bbox_inches='tight')

else:
    print('PLease run this file as level 0')
    exit(1)

