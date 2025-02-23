import argparse
import copy
import csv
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import trange
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR

import sys



from experiments.FLHetero.Models.CNNs import CNN_1_hetero_AFM, CNN_1_homo_AFM
from experiments.FLHetero.Models.trainable_params import vector_alpha
from experiments.FLHetero.node import BaseNodes
from experiments.utils import get_device, set_logger, set_seed, str2bool
from sklearn.decomposition import PCA


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

random.seed(2022)

torch.autograd.set_detect_anomaly(True)


def test_acc(net_large, net_small, net_alpha, testloader,criteria):
    # net.eval()
    with torch.no_grad():
        test_acc = 0
        num_batch = 0

        for batch in testloader:
            num_batch += 1
            # batch = next(iter(testloader))

            img, label = tuple(t.to(device) for t in batch)
            # pred,_ = net(img,torch.zeros(500).to(device),torch.ones(500).to(device)) # img, homo_rep, alpha_vector
            _, small_rep = net_small(img)

            alpha = net_alpha.alpha

            small_rep = net_alpha(small_rep)
            large_pred, mix_rep = net_large(img, small_rep, alpha)

            test_loss = criteria(large_pred, label)
            test_acc += large_pred.argmax(1).eq(label).sum().item() / len(label)
        mean_test_loss = test_loss / num_batch
        mean_test_acc = test_acc / num_batch
    return mean_test_loss, mean_test_acc

def test_acc_global(net_large, net_small, testloader,criteria):
    # net.eval()
    with torch.no_grad():
        test_acc = 0
        num_batch = 0

        for batch in testloader:
            num_batch += 1
            # batch = next(iter(testloader))

            img, label = tuple(t.to(device) for t in batch)
            # pred,_ = net(img,torch.zeros(500).to(device),torch.ones(500).to(device)) # img, homo_rep, alpha_vector
            _, small_rep = net_small(img)

            # alpha = net_alpha.alpha

            # small_rep = net_alpha(small_rep)
            large_pred, mix_rep = net_large(img, small_rep, torch.zeros(500))

            test_loss = criteria(large_pred, label)
            test_acc += large_pred.argmax(1).eq(label).sum().item() / len(label)
        mean_test_loss = test_loss / num_batch
        mean_test_acc = test_acc / num_batch
    return mean_test_loss, mean_test_acc

def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int, fraction: float,
          steps: int, epochs: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int) -> None:

    ###############################
    # init nodes, hnet, local net #
    ###############################
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node,
                      batch_size=bs)

    # -------compute aggregation weights-------------#
    train_sample_count = nodes.train_sample_count
    eval_sample_count = nodes.eval_sample_count
    test_sample_count = nodes.test_sample_count

    client_sample_count = [train_sample_count[i] + eval_sample_count[i] + test_sample_count[i] for i in
                           range(len(train_sample_count))]
    # -----------------------------------------------#


    print(data_name)
    if data_name == "cifar10":
        net_large = CNN_1_hetero_AFM(n_kernels=n_kernels)
        net_small = CNN_1_homo_AFM(n_kernels=n_kernels)
    elif data_name == "cifar100":
        net_large = CNN_1_hetero_AFM(n_kernels=n_kernels, out_dim=100)
        net_small = CNN_1_homo_AFM(n_kernels=n_kernels)
    elif data_name == "mnist":
        net = CNN_1_hetero_AFM(n_kernels=n_kernels)
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")


    net_large = net_large.to(device)
    net_small = net_small.to(device)


    alpha_model = vector_alpha()
    alpha_model = alpha_model.to(device)

    ##################
    # init optimizer #
    ##################

    lr_alpha = 1e-1
    reg_weight = 0.01
    optimizer_small = torch.optim.SGD(params=net_small.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    optimizer_large = torch.optim.SGD(params=net_large.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    optimizer_alpha = torch.optim.SGD(params=alpha_model.parameters(), lr=lr_alpha, momentum=0.9, weight_decay=wd)
    # optimizer_alpha = torch.optim.Adam(alpha_model.parameters(), lr=lr_alpha)

    # times = 2
    # interval = 5
    # scheduler_lambda = lambda epoch: times ** (epoch // interval)

    # scheduler = LambdaLR(optimizer_alpha, lr_lambda=scheduler_lambda)

    criteria = torch.nn.CrossEntropyLoss()
    # scheduler_small = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_small,
    #                                            milestones=[int(steps * 0.56), int(steps * 0.78)],
    #                                            gamma=0.1, last_epoch=-1)
    # scheduler_large = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_large,
    #                                                        milestones=[int(steps * 0.56), int(steps * 0.78)],
    #                                                        gamma=0.1, last_epoch=-1)
    # scheduler_gate = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_gate,
    #                                                        milestones=[int(steps * 0.56), int(steps * 0.78)],
    #                                                        gamma=10, last_epoch=-1)


    ################
    # init metrics #
    ################
    step_iter = trange(steps)

    GM_small = copy.deepcopy(net_small.state_dict())


    PM_large_acc = defaultdict()
    PM_small = defaultdict()
    PM_large = defaultdict()
    PM_gate = defaultdict()
    PM_mix_acc = defaultdict()
    ALPHA = defaultdict()
    Alpha_mean = defaultdict()

    for i in range(num_nodes):
        PM_large_acc[i] = 0
        PM_mix_acc[i] = 0
        PM_small[i] = GM_small
        PM_large[i] = copy.deepcopy(net_large.state_dict())
        ALPHA[i] = torch.ones(500)
        Alpha_mean[i] = torch.mean(ALPHA[i])


    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    orininal_file = str(save_path / f"Homo_FedAFM_globalTest_mixAcc_lr{lr_alpha}_N{num_nodes}_C{fraction}_R{steps}_E{epochs}_B{bs}_NonIID_{data_name}.csv")

    with open(orininal_file, 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')

        for step in step_iter:  # step is round
            frac = fraction
            select_nodes = random.sample(range(num_nodes), int(frac * num_nodes))

            small_local_trained_loss = []
            small_local_trained_acc = []
            small_global_loss = []
            small_global_acc = []
            large_local_trained_loss = []
            large_local_trained_acc = []
            results = []


            LNs = defaultdict() # colloect small_local_model


            logging.info(f'#----Round:{step}----#')
            for c in select_nodes:
                node_id = c

                net_small.load_state_dict(GM_small)
                net_large.load_state_dict(PM_large[node_id])


                alpha_model.alpha.data = torch.tensor(ALPHA[node_id]) # use alpha in the last round


                # evlaute GM
                # global_loss,  global_acc = test_acc(net_small, nodes.test_loaders[node_id], criteria)
                global_loss, global_acc = test_acc_global(net_large, net_small, nodes.test_loaders[node_id], criteria)
                small_global_loss.append(global_loss.cpu().item())
                small_global_acc.append(global_acc)

                for i in range(epochs):
                    # net_small.train()
                    net_large.train()
                    alpha_model.train()
                    for j, batch in enumerate(nodes.train_loaders[node_id]):
                        img, label = tuple(t.to(device) for t in batch)

                        _, small_rep = net_small(img)

                        alpha = alpha_model.alpha


                        small_rep = alpha_model(small_rep)
                        large_pred, mix_rep = net_large(img, small_rep, alpha)


                        loss = criteria(large_pred, label)

                        optimizer_large.zero_grad()
                        # optimizer_small.zero_grad()
                        optimizer_alpha.zero_grad()

                        loss.backward(retain_graph=True)

                        torch.nn.utils.clip_grad_norm_(net_large.parameters(), 50)
                        # torch.nn.utils.clip_grad_norm_(net_small.parameters(), 50)

                        # torch.nn.utils.clip_grad_norm_(alpha_model.parameters(), 1)


                        alpha_model.alpha.data = torch.clamp(alpha_model.alpha.data, 0, 1)

                        optimizer_large.step()
                        # optimizer_small.step()
                        optimizer_alpha.step()
                        alpha_model.alpha.data = torch.clamp(alpha_model.alpha.data, 0, 1)

                # collect local NN parameters

                PM_large[node_id] = copy.deepcopy(net_large.state_dict())
                ALPHA[node_id] = alpha
                Alpha_mean[node_id] = torch.mean(alpha)
                # print(f'alpha:{torch.mean(alpha)}')

                trained_loss, trained_acc = test_acc(net_large, net_small, alpha_model, nodes.test_loaders[node_id], criteria)
                large_local_trained_loss.append(trained_loss.cpu().item())
                large_local_trained_acc.append(trained_acc)
                PM_large_acc[node_id] = trained_acc

                for i in range(epochs):
                    net_small.train()

                    for j, batch in enumerate(nodes.train_loaders[node_id]):
                        img, label = tuple(t.to(device) for t in batch)

                        _, small_rep = net_small(img)

                        # alpha = alpha_model.alpha.item()
                        # small_rep = alpha_model(small_rep)
                        large_pred, large_rep = net_large(img, small_rep, torch.zeros(500))

                        loss = criteria(large_pred, label)

                        optimizer_small.zero_grad()

                        loss.backward(retain_graph=True)

                        torch.nn.utils.clip_grad_norm_(net_small.parameters(), 50)

                        optimizer_small.step()



                LNs[node_id] = net_small.state_dict()
                # evaluate trained local model
                # trained_loss, trained_acc = test_acc(net_small, nodes.test_loaders[node_id], criteria)
                # small_local_trained_loss.append(trained_loss.cpu().item())
                # small_local_trained_acc.append(trained_acc)



            # scheduler.step()
            # lr_alpha = optimizer_alpha.param_groups[0]['lr']
            # print(f'current alpha learning rate:{lr_alpha}')
            #
            # if lr_alpha > 0.1:
            #     for param_group in optimizer_alpha.param_groups:
            #         param_group['lr'] = 0.1


            # scheduler_small.step()
            # scheduler_large.step()
            # scheduler_gate.step()
            # LR_gate[node_id] = scheduler_gate.get_last_lr()scheduler_gate
            # print('\t last_lr:', scheduler_gate.get_last_lr())

            mean_small_global_loss = round(np.mean(small_global_loss), 4)
            mean_small_global_acc = round(np.mean(small_global_acc), 4)
            # mean_small_trained_loss = round(np.mean(small_local_trained_loss), 4)
            # mean_small_trained_acc = round(np.mean(small_local_trained_acc), 4)
            mean_large_trained_loss = round(np.mean(large_local_trained_loss), 4)
            mean_large_trained_acc = round(np.mean(large_local_trained_acc), 4)



            # results = [mean_small_global_loss, mean_small_global_acc, mean_small_trained_loss, mean_small_trained_acc, mean_large_trained_loss, mean_large_trained_acc] + [round(i) for i in PM_large_acc.values()]
            results = [mean_small_global_loss, mean_small_global_acc, mean_large_trained_loss, mean_large_trained_acc] + [round(i,4) for i in PM_large_acc.values()] + [i.data.item() for i in Alpha_mean.values()]  #+ [i for i in LR_gate.values()]
            print(f'Round {step} | Mean Large Acc: {mean_large_trained_acc}')


            mywriter.writerow(results)
            file.flush()
            # logging.info(
            #     f'Round:{step} | small_GM_loss:{mean_small_global_loss} | small_PM_loss:{mean_small_trained_loss} | large_PM_loss:{mean_large_trained_loss}')
            # logging.info(
            #     f'Round:{step} | small_GM_acc:{mean_small_global_acc} | small_PM_acc:{mean_small_trained_acc} | large_PM_acc:{mean_large_trained_acc}')

            client_agg_weights = OrderedDict()
            select_nodes_sample_count = OrderedDict()
            for i in range(len(select_nodes)):
                select_nodes_sample_count[select_nodes[i]] = client_sample_count[select_nodes[i]]
            for i in range(len(select_nodes)):
                client_agg_weights[select_nodes[i]] = select_nodes_sample_count[select_nodes[i]] / sum(select_nodes_sample_count.values())


            weight_keys = list(net_small.state_dict().keys())
            for key in weight_keys:
                key_sum = 0
                for id, model in LNs.items():
                    key_sum += client_agg_weights[id] * model[key]
                GM_small[key] = key_sum
            logging.info(f'Global model is updated after aggregation')

        logging.info('Federated Learning has been successfully!')

    new_file = str(save_path / f"Done_Homo_FedAFM_globalTest_mixAcc_lr{lr_alpha}_N{num_nodes}_C{fraction}_R{steps}_E{epochs}_B{bs}_NonIID_{data_name}.csv")

    os.rename(orininal_file, new_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Learning with Lookahead experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100', 'mnist'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
    parser.add_argument("--num-nodes", type=int, default=50, help="number of simulated nodes")
    parser.add_argument("--fraction", type=int, default=0.2, help="number of sampled nodes in each round")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=512) # 512
    parser.add_argument("--epochs", type=int, default=10, help="number of inner steps")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-3, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="Results/FedAFM", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if args.data_name == 'cifar10':
        args.classes_per_node = 2
    elif args.data_name == 'cifar100':
        args.classes_per_node = 10
    else:
        args.classes_per_node = 2

    train(
        data_name=args.data_name,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        num_nodes=args.num_nodes,
        fraction=args.fraction,
        steps=args.num_steps,
        epochs=args.epochs,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed
    )
