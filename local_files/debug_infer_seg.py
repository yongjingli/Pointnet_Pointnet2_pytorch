import os
import shutil
import numpy as np
import h5py
import torch
import argparse
import importlib
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from tqdm import tqdm
import sys

sys.path.insert(0, "../")
os.chdir("../")

sys.path.append(os.path.join('./', 'models'))


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


def debug_infer_seg(args):
    save_path = "/userdata/liyj/data/test_data/depth/debug"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    # data_path = 'data/modelnet40_normal_resampled/'
    # test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    # DataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    # load model
    experiment_dir = 'log/sem_seg/' + args.log_dir
    num_class = 13
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    xyz = torch.rand(1, 9, 2048)
    if not args.use_cpu:
        xyz = xyz.cuda()

    # xyz.cuda()
    seg_pred, _ = classifier(xyz)
    print(seg_pred.shape)

    #
    # vote_num = 1
    # for j, (points, target) in tqdm(enumerate(DataLoader), total=len(DataLoader)):
    #     vote_pool = torch.zeros(target.size()[0], num_class).cuda()
    #     print("vote_pool:", vote_pool.shape)
    #     if not args.use_cpu:
    #         points, target = points.cuda(), target.cuda()
    #
    #     points = points.transpose(2, 1)
    #     print("points:", points.shape)
    #
    #     ply_path = os.path.join(save_path, str(j) + ".ply")
    #     save_ply(points.transpose(2, 1)[0], ply_path)
    #
    #     for _ in range(vote_num):
    #         seg_pred, _ = classifier(points)
    #         # print("pred:", pred.shape)
    #         # print(pred)
    #     #     vote_pool += pred
    #     # pred = vote_pool / vote_num
    #     # pred_choice = pred.data.max(1)[1]
    #     # print(pred_choice)
    #     exit(1)


if __name__ == "__main__":
    print("Start...")
    args = parse_args()
    args.log_dir = "pointnet2_sem_seg"
    args.use_cpu = True
    debug_infer_seg(args)
    print("End...")