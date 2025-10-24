import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import json
import argparse
from inline_stab_sgd import InlineStabSGD
from models import resnet56_cifar


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(loader, model, criterion):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    correct_count = 0

    with torch.no_grad():
        for (data, target) in loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)

            class_pred = output.data.max(1, keepdim=True)[1]
            correct_count += class_pred.eq(target.data.view_as(class_pred)).sum().item()
    
    accuracy = 100. * correct_count / len(loader.dataset)
    return total_loss / len(loader.dataset), accuracy


def train_model(train_loader, test_loader, model, criterion, optimizer, min_gd_iterations, min_total_iterations, output_folder, args):
    device = next(model.parameters()).device

    init_train_loss, init_train_acc = evaluate_model(train_loader, model, criterion)
    init_test_loss, init_test_acc = evaluate_model(test_loader, model, criterion)

    gd_test_acc_evol = [(0, init_test_acc)]
    total_test_acc_evol = [(0, init_test_acc)]

    gd_test_loss_evol = [(0, init_test_loss)]
    total_test_loss_evol = [(0, init_test_loss)]

    gd_train_acc_evol = [(0, init_train_acc)]
    total_train_acc_evol = [(0, init_train_acc)]

    gd_train_loss_evol = [(0, init_train_loss)]
    total_train_loss_evol = [(0, init_train_loss)]

    
    print("\n \n --- Pre-train eval ---")
    print(f'Normalized eval loss: {init_test_loss:.3e}')
    print(f'Normalized train loss: {init_train_loss:.3e}')
    print(f'Test acc: {init_test_acc:.3f}%')
    print("\n \n")
    

    # If a multiple of len(train_loader) is during Stab computation time we would compute the Stab ratio a zillion times...
    already_evaluated_gd = []

    epoch = 0
    gd_training_complete = False
    total_training_complete = False
    training_complete = False
    
    while not training_complete:
        # Train
        model.train()
        
        for data, target in train_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            gd_iteration = optimizer.state_dict()["state"]["param"]["t_gd"]
            if gd_iteration >= min_gd_iterations:
                gd_training_complete = True

            total_iteration = optimizer.state_dict()["state"]["param"]["t"]
            if total_iteration >= min_total_iterations:
                total_training_complete = True
            

            print(f"GD Iteration: {gd_iteration}, Total iteration: {total_iteration}", end='\r')

            training_complete = gd_training_complete and total_training_complete
            if training_complete:
                break

            
            # Evaluate model for "gradient descent epochs"
            if gd_iteration % len(train_loader) == 0 and gd_iteration > 0 and gd_iteration not in already_evaluated_gd:
                print(f"\n Currently evalutating GD iteration {gd_iteration}")
                already_evaluated_gd.append(gd_iteration)

                train_loss, train_accuracy = evaluate_model(train_loader, model, criterion)
                test_loss, test_accuracy = evaluate_model(test_loader, model, criterion)

                gd_test_acc_evol.append((gd_iteration, test_accuracy))
                gd_train_acc_evol.append((gd_iteration, train_accuracy))

                gd_test_loss_evol.append((gd_iteration, test_loss))
                gd_train_loss_evol.append((gd_iteration, train_loss))

                print(f'Current train loss: {train_loss:.3e}')
                print(f'Current test accuracy: {test_accuracy}%')
                
                model.train()

        if training_complete:
            break

        # Evaluate model for "real epochs"
        total_iteration = optimizer.state_dict()["state"]["param"]["t"]
        print(f"\n Currently evaluating total iteration: {total_iteration}")

        train_loss, train_accuracy = evaluate_model(train_loader, model, criterion)
        test_loss, test_accuracy = evaluate_model(test_loader, model, criterion)

        total_test_acc_evol.append((total_iteration, test_accuracy))
        total_train_acc_evol.append((total_iteration, train_accuracy))

        total_test_loss_evol.append((total_iteration, test_loss))
        total_train_loss_evol.append((total_iteration, train_loss))

        print(f'Current train loss: {train_loss:.3e}')
        print(f'Current test accuracy: {test_accuracy}%')

        epoch += 1
        print(f'Epoch {epoch} completed')


    # Evalute at the end of training
    total_iteration = optimizer.state_dict()["state"]["param"]["t"]
    gd_iteration = optimizer.state_dict()["state"]["param"]["t_gd"]

    print(f"Final evaluation at total iteration: {total_iteration}, gd iteration: {gd_iteration}")

    final_train_loss, final_train_acc = evaluate_model(train_loader, model, criterion)
    final_test_loss, final_test_acc = evaluate_model(test_loader, model, criterion)

    gd_test_acc_evol.append((gd_iteration, final_test_acc))
    total_test_acc_evol.append((total_iteration, final_test_acc))

    gd_test_loss_evol.append((gd_iteration, final_test_loss))
    total_test_loss_evol.append((total_iteration, final_test_loss))

    gd_train_acc_evol.append((gd_iteration, final_train_acc))
    total_train_acc_evol.append((total_iteration, final_train_acc))

    gd_train_loss_evol.append((gd_iteration, final_train_loss))
    total_train_loss_evol.append((total_iteration, final_train_loss))



    total_train_acc_evol = np.array(total_train_acc_evol)
    total_test_acc_evol = np.array(total_test_acc_evol)
    total_train_loss_evol = np.array(total_train_loss_evol)
    total_test_loss_evol = np.array(total_test_loss_evol)

    with open(os.path.join(output_folder, 'total_train_acc_evol.npy'), 'wb') as f:
        np.save(f, total_train_acc_evol)
    with open(os.path.join(output_folder, 'total_test_acc_evol.npy'), 'wb') as f:
        np.save(f, total_test_acc_evol)
    with open(os.path.join(output_folder, 'total_train_loss_evol.npy'), 'wb') as f:
        np.save(f, total_train_loss_evol)
    with open(os.path.join(output_folder, 'total_test_loss_evol.npy'), 'wb') as f:
        np.save(f, total_test_loss_evol)
    
    gd_train_acc_evol = np.array(gd_train_acc_evol)
    gd_test_acc_evol = np.array(gd_test_acc_evol)
    gd_train_loss_evol = np.array(gd_train_loss_evol)
    gd_test_loss_evol = np.array(gd_test_loss_evol)

    with open(os.path.join(output_folder, 'gd_train_acc_evol.npy'), 'wb') as f:
        np.save(f, gd_train_acc_evol)
    with open(os.path.join(output_folder, 'gd_test_acc_evol.npy'), 'wb') as f:
        np.save(f, gd_test_acc_evol)
    with open(os.path.join(output_folder, 'gd_train_loss_evol.npy'), 'wb') as f:
        np.save(f, gd_train_loss_evol)
    with open(os.path.join(output_folder, 'gd_test_loss_evol.npy'), 'wb') as f:
        np.save(f, gd_test_loss_evol)

    # Save hyperparameters and optimizer info
    hyperparameters = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "min_gd_iterations": min_gd_iterations,
        "min_total_iterations": min_total_iterations,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "zeta_start": args.zeta_start,
        "zeta": args.zeta,
        "kappa": args.kappa,
        "gamma": args.gamma
    }

    optimizer_info = {
        "optimizer_type": type(optimizer).__name__,
        "defaults": optimizer.defaults 
    }
    
    optimizer_info["stab_history"] = optimizer.stab_history
    optimizer_info["kurtosis_history"] = optimizer.kurtosis_history
    optimizer_info["total_samples"] = optimizer.total_compute_samples 
    
    hyperparameters["optimizer"] = optimizer_info


    # Save to a JSON file
    with open(os.path.join(output_folder, "hyperparameters.json"), "w") as f:
        json.dump(hyperparameters, f, indent=4)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=10.0)
    parser.add_argument('--min-gd-iterations', type=int, default=64000, help='Minimum number of gradient descent iterations')
    parser.add_argument('--min-total-iterations', type=int, default=128000, help='Minimum number of total iterations')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--zeta-start', type=int, default=100.0, help='Stab ratio checkpoint start')
    parser.add_argument('--zeta', type=float, default=100.0, help='Stab ratio checkpoint factor')
    parser.add_argument('--kappa', type=float, default=0.1, help='Stab ratio checkpoint factor')
    parser.add_argument('--gamma', type=float, default=1, help='Stab ratio checkpoint exponent')
    parser.add_argument('--output-folder', type=str, default="results", help='Output folder for results')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build output path
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    model = resnet56_cifar().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = InlineStabSGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        zeta_start=args.zeta_start,
        zeta=args.zeta,
        kappa=args.kappa,
        gamma=args.gamma
    )

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    def seed_worker(worker_id):
        np.random.seed(args.seed + worker_id)
        torch.manual_seed(args.seed + worker_id)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                             pin_memory=True, worker_init_fn=seed_worker, generator=g)

    print(f"Using device: {device}")
    print(f"Training until {args.min_gd_iterations} gradient descent iterations and {args.min_total_iterations} total iterations with lr={args.lr}, seed={args.seed}, batch size={args.batch_size}")

    train_model(train_loader, test_loader, model, criterion, optimizer, args.min_gd_iterations, args.min_total_iterations, output_folder, args)
    print("Training complete.")