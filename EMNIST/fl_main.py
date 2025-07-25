from GLFC import GLFC_model
from ResNet import resnet18_cbam
import torch
import copy
import random
import os.path as osp
import os
from myNetwork import network, LeNet
from Fed_utils import * 
from ProxyServer import * 
from mini_imagenet import *
from tiny_imagenet import *
from option import args_parser

# Define the modified functions
def local_train_with_classes(clients, index, model_g, task_id, model_old, ep_g, old_client):
    clients[index].model = copy.deepcopy(model_g)
    
    # Set current classes based on client's task assignment
    if task_id < len(clients[index].client_classes):
        clients[index].current_class = clients[index].client_classes[task_id]
    else:
        # If task_id exceeds client's tasks, use last task
        clients[index].current_class = clients[index].client_classes[-1]

    if index in old_client:
        clients[index].beforeTrain(task_id, 0)
    else:
        clients[index].beforeTrain(task_id, 1)

    clients[index].update_new_set()
    print(f'Client {index} training on classes: {clients[index].current_class}')
    clients[index].train(ep_g, model_old)
    local_model = clients[index].model.state_dict()
    proto_grad = clients[index].proto_grad_sharing()

    print('*' * 60)

    return local_model, proto_grad

def participant_exemplar_storing_with_classes(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        clients[index].model = copy.deepcopy(model_g)
        
        # Set current classes based on client's task assignment
        if task_id < len(clients[index].client_classes):
            clients[index].current_class = clients[index].client_classes[task_id]
        else:
            clients[index].current_class = clients[index].client_classes[-1]
            
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0)
            else:
                clients[index].beforeTrain(task_id, 1)
            clients[index].update_new_set()

args = args_parser()

## parameters for learning
feature_extractor = resnet18_cbam()
num_clients = 10  # Changed from args.num_clients to 10
old_client_0 = []
old_client_1 = [i for i in range(10)]  # Changed to 10 clients
new_client = []
models = []

## seed settings
setup_seed(args.seed)

## model settings
# Initialize with 100 classes since we're using CIFAR-100
initial_classes = 100 if args.dataset == 'cifar100' else args.numclass
model_g = network(initial_classes, feature_extractor)
model_g = model_to_device(model_g, False, args.device)
model_old = None

train_transform = transforms.Compose([transforms.RandomCrop((args.img_size, args.img_size), padding=4),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.24705882352941178),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
test_transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), 
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

if args.dataset == 'cifar100':
    train_dataset = iCIFAR100('dataset', transform=train_transform, download=True)
    test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=True)

elif args.dataset == 'tiny_imagenet':
    train_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform, test_transform=test_transform)
    train_dataset.get_data()
    test_dataset = train_dataset

else:
    train_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
    train_dataset.get_data()
    test_dataset = train_dataset

encode_model = LeNet(num_classes=100)
encode_model.apply(weights_init)

# Create clients with predefined class assignments
# Each client has 4 tasks with 20 classes per task
all_classes = list(range(100))
random.shuffle(all_classes)  # Shuffle classes for random assignment

client_classes = {}
classes_per_client = 80  # 4 tasks * 20 classes per task
classes_per_task = 20

# Assign classes to clients with overlap
for i in range(10):
    # Each client gets 80 classes (some will overlap between clients)
    start_idx = (i * 40) % 100  # This creates overlap between clients
    client_classes[i] = []
    for j in range(classes_per_client):
        class_idx = (start_idx + j) % 100
        client_classes[i].append(class_idx)
    # Shuffle client's classes for random task assignment
    random.shuffle(client_classes[i])
    
    # Divide into 4 tasks
    client_classes[i] = [
        client_classes[i][j*classes_per_task:(j+1)*classes_per_task] 
        for j in range(4)
    ]

# Initialize client models
for i in range(10):
    model_temp = GLFC_model(initial_classes, feature_extractor, args.batch_size, 
                           classes_per_task, args.memory_size,  # task_size is now 20
                           args.epochs_local, args.learning_rate, train_dataset, 
                           args.device, encode_model)
    # Store the class assignment for this client
    model_temp.client_classes = client_classes[i]
    model_temp.client_id = i
    models.append(model_temp)

## the proxy server
proxy_server = proxyServer(args.device, args.learning_rate, initial_classes, feature_extractor, encode_model, train_transform)

## training log
output_dir = osp.join('./training_log', args.method, 'seed' + str(args.seed))
if not osp.exists(output_dir):
    os.system('mkdir -p ' + output_dir)
if not osp.exists(output_dir):
    os.mkdir(output_dir)

out_file = open(osp.join(output_dir, 'log_tar_' + str(classes_per_task) + '_10clients.txt'), 'w')
log_str = 'method_{}, task_size_{}, num_clients_10, learning_rate_{}'.format(args.method, classes_per_task, args.learning_rate)
out_file.write(log_str + '\n')
out_file.flush()

classes_learned = initial_classes  # Start with all classes for CIFAR-100
old_task_id = -1
num_tasks = 4  # Each client has 4 tasks

for ep_g in range(args.epochs_global):
    pool_grad = []
    model_old = proxy_server.model_back()
    task_id = ep_g // args.tasks_global

    # No need to add new clients since we have fixed 10 clients
    if task_id != old_task_id and old_task_id != -1:
        # Keep all 10 clients throughout training
        overall_client = 10
        old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.9))
        old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
        num_clients = 10
        print(old_client_0)

    if task_id != old_task_id and old_task_id != -1 and task_id < num_tasks:
        # No need to increment classes_learned since model already handles all 100 classes
        # Just update the model architecture if needed
        pass
    
    print('federated global round: {}, task_id: {}'.format(ep_g, task_id))

    w_local = []
    clients_index = random.sample(range(num_clients), min(args.local_clients, num_clients))
    print('select part of clients to conduct local training')
    print(clients_index)

    for c in clients_index:
        # Use the modified local_train function
        local_model, proto_grad = local_train_with_classes(models, c, model_g, task_id, model_old, ep_g, old_client_0)
        w_local.append(local_model)
        if proto_grad != None:
            for grad_i in proto_grad:
                pool_grad.append(grad_i)

    ## every participant save their current training data as exemplar set
    print('every participant start updating their exemplar set and old model...')
    participant_exemplar_storing_with_classes(models, num_clients, model_g, old_client_0, task_id, clients_index)
    print('updating finishes')

    print('federated aggregation...')
    w_g_new = FedAvg(w_local)
    w_g_last = copy.deepcopy(model_g.state_dict())
    
    model_g.load_state_dict(w_g_new)

    proxy_server.model = copy.deepcopy(model_g)
    proxy_server.dataloader(pool_grad)

    # Evaluate on all classes seen so far
    acc_global = model_global_eval(model_g, test_dataset, min(task_id, num_tasks-1), classes_per_task, args.device)
    log_str = 'Task: {}, Round: {} Accuracy = {:.2f}%'.format(task_id, ep_g, acc_global)
    out_file.write(log_str + '\n')
    out_file.flush()
    print('classification accuracy of global model at round %d: %.3f \n' % (ep_g, acc_global))

    old_task_id = task_id