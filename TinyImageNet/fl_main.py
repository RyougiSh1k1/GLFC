from GLFC import GLFC_model
from ResNet import resnet18_cbam_grayscale
import torch
import copy
import random
import os.path as osp
import os
from myNetwork import network, LeNet
from Fed_utils import * 
from ProxyServer import proxyServer
from iEMNIST import iEMNIST
from option import args_parser
from logger import TrainingLogger
import time

# Define the modified functions
def local_train_with_classes(clients, index, model_g, task_id, model_old, ep_g, old_client, logger):
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
    
    logger.log_client(index, f'Training on classes: {clients[index].current_class}, Signal: {clients[index].signal}')
    
    start_time = time.time()
    clients[index].train(ep_g, model_old)
    train_time = time.time() - start_time
    
    logger.log_client(index, f'Training completed in {train_time:.2f} seconds')
    
    local_model = clients[index].model.state_dict()
    proto_grad = clients[index].proto_grad_sharing()

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

# Parse arguments
args = args_parser()

# Update arguments for EMNIST
args.dataset = 'emnist'
args.img_size = 28  # EMNIST images are 28x28
args.num_clients = 8  # 8 clients
args.local_clients = 4  # Select 4 clients per round
args.task_size = 4  # 4 classes per task
args.numclass = 4  # Initial number of classes
args.memory_size = 500  # Smaller memory for EMNIST

## parameters for learning
feature_extractor = resnet18_cbam_grayscale()
num_clients = 8
old_client_0 = []
old_client_1 = [i for i in range(8)]
new_client = []
models = []

## seed settings
setup_seed(args.seed)

# Initialize logger
log_dir = osp.join('./training_log', args.method, 'emnist', f'seed{args.seed}')
logger = TrainingLogger(log_dir, 'emnist_training')
logger.log_config(args)

## model settings
# EMNIST byclass has 62 classes (10 digits + 26 lowercase + 26 uppercase)
total_classes = 62
initial_classes = total_classes
model_g = network(initial_classes, feature_extractor)
model_g = model_to_device(model_g, False, args.device)
model_old = None

# Transforms for EMNIST (grayscale images)
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # Small rotation for augmentation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Single channel normalization
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load EMNIST dataset
logger.info("Loading EMNIST dataset...")
train_dataset = iEMNIST('dataset', split='byclass', train=True, transform=train_transform, download=True)
test_dataset = iEMNIST('dataset', split='byclass', train=False, test_transform=test_transform, download=True)
logger.info(f"Dataset loaded. Total classes: {total_classes}")

# Modify encode_model for EMNIST (grayscale input)
encode_model = LeNet(channel=1, hideen=588, num_classes=total_classes)  # 1 channel for grayscale
encode_model.apply(weights_init)

# Create clients with predefined class assignments
# Each client has 6 tasks with 4 classes per task
all_classes = list(range(total_classes))
random.shuffle(all_classes)  # Shuffle classes for random assignment

client_classes = {}
tasks_per_client = 6
classes_per_task = 4

# Assign classes to clients with overlap
# We need to ensure all classes are covered and create some overlap
logger.info("Assigning classes to clients...")
for i in range(num_clients):
    client_classes[i] = []
    # Each client gets 24 classes (6 tasks * 4 classes)
    # We'll create overlapping assignments
    client_class_pool = []
    
    # Start with a base assignment and add random classes
    base_start = (i * 20) % total_classes
    for j in range(30):  # Get more classes than needed to allow shuffling
        class_idx = (base_start + j) % total_classes
        client_class_pool.append(class_idx)
    
    # Add some random classes for diversity
    additional_classes = random.sample(all_classes, 10)
    client_class_pool.extend(additional_classes)
    
    # Remove duplicates and shuffle
    client_class_pool = list(set(client_class_pool))
    random.shuffle(client_class_pool)
    
    # Select 24 classes for this client
    selected_classes = client_class_pool[:24]
    
    # Divide into 6 tasks
    for task in range(tasks_per_client):
        task_classes = selected_classes[task*classes_per_task:(task+1)*classes_per_task]
        client_classes[i].append(task_classes)
    
    logger.info(f"Client {i} assigned tasks: {client_classes[i]}")

# Initialize client models
logger.info("Initializing client models...")
for i in range(num_clients):
    model_temp = GLFC_model(initial_classes, feature_extractor, args.batch_size, 
                           classes_per_task, args.memory_size, args.epochs_local, 
                           args.learning_rate, train_dataset, args.device, encode_model)
    # Store the class assignment for this client
    model_temp.client_classes = client_classes[i]
    model_temp.client_id = i
    models.append(model_temp)

## the proxy server
proxy_server = proxyServer(args.device, args.learning_rate, initial_classes, feature_extractor, encode_model, train_transform)

## training log file
out_file = open(osp.join(log_dir, f'results_8clients_6tasks_4classes.txt'), 'w')
log_str = f'method_{args.method}, task_size_{classes_per_task}, num_clients_{num_clients}, tasks_per_client_{tasks_per_client}, learning_rate_{args.learning_rate}'
out_file.write(log_str + '\n')
out_file.write('='*80 + '\n')
out_file.flush()

classes_learned = initial_classes
old_task_id = -1
num_tasks = tasks_per_client
best_accuracy = 0.0

logger.info("Starting federated training...")
logger.info(f"Total rounds: {args.epochs_global}, Tasks per client: {num_tasks}")

for ep_g in range(args.epochs_global):
    round_start_time = time.time()
    pool_grad = []
    model_old = proxy_server.model_back()
    task_id = ep_g // args.tasks_global

    # Keep all clients throughout training
    if task_id != old_task_id and old_task_id != -1:
        old_client_1 = random.sample([i for i in range(num_clients)], int(num_clients * 0.9))
        old_client_0 = [i for i in range(num_clients) if i not in old_client_1]
        logger.info(f"New clients for task {task_id}: {old_client_0}")

    logger.info(f'Federated global round: {ep_g}, Task ID: {task_id}')

    w_local = []
    clients_index = random.sample(range(num_clients), min(args.local_clients, num_clients))
    logger.info(f'Selected clients for training: {clients_index}')

    # Local training for selected clients
    for c in clients_index:
        local_model, proto_grad = local_train_with_classes(models, c, model_g, task_id, model_old, ep_g, old_client_0, logger)
        w_local.append(local_model)
        if proto_grad != None:
            for grad_i in proto_grad:
                pool_grad.append(grad_i)

    # Update exemplar sets for all clients
    logger.info('Updating exemplar sets for all clients...')
    participant_exemplar_storing_with_classes(models, num_clients, model_g, old_client_0, task_id, clients_index)

    # Federated aggregation
    logger.info('Performing federated aggregation...')
    w_g_new = FedAvg(w_local)
    model_g.load_state_dict(w_g_new)

    # Update proxy server
    proxy_server.model = copy.deepcopy(model_g)
    proxy_server.dataloader(pool_grad)

    # Evaluate global model
    acc_global = model_global_eval(model_g, test_dataset, min(task_id, num_tasks-1), classes_per_task, args.device)
    
    round_time = time.time() - round_start_time
    
    # Log metrics
    metrics = {
        'accuracy': acc_global,
        'round_time': round_time,
        'num_clients_trained': len(clients_index),
        'pool_grad_size': len(pool_grad)
    }
    logger.log_round(ep_g, task_id, metrics)
    
    # Update best accuracy
    if acc_global > best_accuracy:
        best_accuracy = acc_global
        logger.info(f'New best accuracy: {best_accuracy:.2f}%')
    
    # Write to output file
    log_str = f'Task: {task_id}, Round: {ep_g}, Accuracy: {acc_global:.2f}%, Time: {round_time:.2f}s, Best: {best_accuracy:.2f}%'
    out_file.write(log_str + '\n')
    out_file.flush()

    old_task_id = task_id

# Final summary
logger.info('='*80)
logger.info('Training completed!')
logger.info(f'Best accuracy achieved: {best_accuracy:.2f}%')
logger.info(f'Log files saved to: {log_dir}')

out_file.write('='*80 + '\n')
out_file.write(f'Training completed. Best accuracy: {best_accuracy:.2f}%\n')
out_file.close()
logger.close()