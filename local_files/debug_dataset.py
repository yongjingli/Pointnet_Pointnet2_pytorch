import os
import shutil
import numpy as np
import h5py
import torch
import argparse
from data_utils.ModelNetDataLoader import ModelNetDataLoader

import sys

sys.path.insert(0, "../")
os.chdir("../")


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=False, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def save_ply(points, ply_path):
    out_points = []
    for point in points:
        out_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2],  255, 255, 255))

    file = open(ply_path, "w")
    file.write('''ply
                    format ascii 1.0
                    element vertex %d
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    property uchar alpha
                    end_header
                    %s
                    ''' % (len(out_points), "".join(out_points)))
    file.close()


def debug_model_net_dataset(args):
    data_path = 'data/modelnet40_normal_resampled/'
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)

    save_path = "/userdata/liyj/data/test_data/depth/debug"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    DataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    for i, (point, label) in enumerate(DataLoader):
        ply_path = os.path.join(save_path, str(i) + ".ply")
        save_ply(point[0], ply_path)
        print(label)
        print(point.shape)
        print(label.shape)
        exit(1)


if __name__ == "__main__":
    print("Start...")
    args = parse_args()
    debug_model_net_dataset(args)
    print("End...")