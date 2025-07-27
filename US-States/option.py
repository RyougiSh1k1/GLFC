import argparse
import torch

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='emnist', help="name of dataset")
    parser.add_argument('--method', type=str, default='glfc', help="name of method")
    parser.add_argument('--iid_level', type=int, default=4, help='number of data classes for local clients')
    parser.add_argument('--numclass', type=int, default=4, help="number of data classes in the first task")
    parser.add_argument('--img_size', type=int, default=28, help="size of images")
    parser.add_argument('--device', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--batch_size', type=int, default=128, help='size of mini-batch')
    parser.add_argument('--task_size', type=int, default=4, help='number of data classes each task')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--memory_size', type=int, default=500, help='size of exemplar memory')
    parser.add_argument('--epochs_local', type=int, default=30, help='local epochs of each global round')
    parser.add_argument('--learning_rate', type=float, default=2.0, help='learning rate')
    parser.add_argument('--num_clients', type=int, default=8, help='initial number of clients')
    parser.add_argument('--local_clients', type=int, default=8, help='number of selected clients each round')
    parser.add_argument('--epochs_global', type=int, default=36, help='total number of global rounds (6 tasks * 6 rounds per task)')
    parser.add_argument('--tasks_global', type=int, default=6, help='rounds per task')
    args = parser.parse_args()
    return args