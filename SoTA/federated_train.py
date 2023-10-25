import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random

from model import *
from utils import *
import scipy
from scipy import stats
from scipy.special import softmax




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple-cnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='moon',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=5, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100,
                        help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None,
                        help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    args = parser.parse_args()
    return args


def getName(model):
    list_name=[]



    for name in model.state_dict():
        list_name.append(name)



    return list_name


def save_model_hos(model,list_name,client):
    np_path='./SaveModel/Model_'+str(client)+'/'


    if os.path.isdir(np_path)==False:
        os.makedirs(np_path)

    for name in list_name:

        temp_num=model.state_dict()[name].cpu().numpy()
        np.save(np_path+"%s.ndim"%(name),temp_num)





def cumulants_layer_prune(temp_load_numpy, n_clients, temp_filter_cumulants,coef):
    if temp_load_numpy[0].ndim == 4 or temp_load_numpy[0].ndim == 2:

        if temp_load_numpy[0].ndim == 4:
            sz=temp_load_numpy[0].shape[0]*temp_load_numpy[0].shape[1]*temp_load_numpy[0].shape[2]*temp_load_numpy[0].shape[3]
        else:
            sz = temp_load_numpy[0].shape[0] * temp_load_numpy[0].shape[1]

        x=np.linspace(-1,1,sz)
        prob_g = stats.norm.pdf(x, 0, 1)

        cumulants_temp = []
        for client in range(n_clients):
            if coef=='KL':

                prob = softmax(temp_load_numpy[client].flatten())


                cumulants_temp.append(np.abs(stats.entropy(prob+1e-10,prob_g+1e-10)))






            elif coef=='R0':

                alpha=0
                prob = softmax(temp_load_numpy[client].flatten())


                cumulants_temp.append(np.abs( (1 / (alpha - 1)) * math.log(sum([prob[i] ** alpha * prob_g[i] ** (1 - alpha) for i in range(len(prob))]))))


            elif coef=='BH':

                alpha=1/2
                prob = softmax(temp_load_numpy[client].flatten())


                cumulants_temp.append(np.abs( (1 / (alpha - 1)) * math.log(sum([prob[i] ** alpha * prob_g[i] ** (1 - alpha) for i in range(len(prob))]))))

            elif coef=='R2':

                alpha=2
                prob = softmax(temp_load_numpy[client].flatten())


                cumulants_temp.append(np.abs( (1 / (alpha - 1)) * math.log(sum([prob[i] ** alpha * prob_g[i] ** (1 - alpha) for i in range(len(prob))]))))


            else:
                cumulants_temp.append(abs(stats.kstat(temp_load_numpy[client],4) * stats.kstat(temp_load_numpy[client], 3)))



        temp_filter_cumulants = cumulants_temp


        temp = np.asarray(temp_load_numpy)
        temp_load_numpy_server = np.average(temp, 0, temp_filter_cumulants)





    elif temp_load_numpy[0].ndim == 1 or temp_load_numpy[0].ndim == 0:

        temp = np.asarray(temp_load_numpy)
        temp_load_numpy_server = np.average(temp, 0, temp_filter_cumulants)

    return temp_load_numpy_server, temp_filter_cumulants



def cumulants_layer_prune_fl(temp_load_numpy, n_clients):

    cumulants_temp = []
    for client in range(n_clients):
        cumulants_temp.append(abs(stats.kstat(temp_load_numpy[client],4) * stats.kstat(temp_load_numpy[client], 3)))
    max_value = max(cumulants_temp)

    weight_avg = cumulants_temp / max_value



    return  weight_avg






def load_model_hos(model, list_name, n_clients, aggregation_method, coef):
    np_path = './SaveModel2/Model_'

    s = 0
    weight_avg = 0
    weight_avg_f=0
    weight_avg_l=0

    for var_name in list_name:
        temp_load_numpy = []

        temp_load_numpy.append(np.load(np_path + '0' + "/%s.ndim.npy" % (var_name)))

        if temp_load_numpy[0].ndim == 4 or temp_load_numpy[0].ndim == 2:
            s += 1

    for var_name in list_name:
        temp_load_numpy = []

        print(var_name)

        for client in range(n_clients):
            temp_load_numpy.append(np.load(np_path + str(client) + "/%s.ndim.npy" % (var_name)))

        if aggregation_method == 'avg':

            temp = np.asarray(temp_load_numpy)
            temp_load_numpy_server = np.mean(temp, 0)


        elif aggregation_method == 'cumulants_layer_prune':

            temp_load_numpy_server, weight_avg = cumulants_layer_prune(temp_load_numpy, n_clients, weight_avg,coef)







            #weight_avg = weight_avg_l * weight_avg_f
            weight_avg_l=np.expand_dims(weight_avg_l, axis=0)
            weight_avg_l=np.resize(weight_avg_l,(weight_avg_f.shape[0],weight_avg_f.shape[1]))


            if len(coef)>1:

                weight_avg=coef[0]*weight_avg_l+coef[1]*weight_avg_f


            else:

                weight_avg = weight_avg_l * weight_avg_f


                if coef=='norm':

                    m=np.amax(weight_avg,axis=0)

                    weight_avg=weight_avg/m



            temp = np.asarray(temp_load_numpy)

            temp_load_numpy_server = np.empty_like(temp_load_numpy[0])

            if temp_load_numpy[0].ndim == 4:

                for i in range(np.shape(temp_load_numpy[0])[0]):
                    temp_load_numpy_server[i, :, :, :] = np.average(temp[:, i, :, :, :], 0, weight_avg[i,:])



            elif temp_load_numpy[0].ndim == 2:

                for i in range(np.shape(temp_load_numpy[0])[0]):
                    temp_load_numpy_server[i, :] = np.average(temp[:, i, :], 0, weight_avg[i,:])





            elif temp_load_numpy[0].ndim == 1:

                for i in range(np.shape(temp_load_numpy[0])[0]):
                    temp_load_numpy_server[i] = np.average(temp[:, i], 0, weight_avg[i,:])


            elif temp_load_numpy[0].ndim == 0:



                temp_load_numpy_server = np.average(temp, 0, np.average(weight_avg,0))

        tensor_load = torch.tensor(temp_load_numpy_server)

        model.state_dict()[var_name].copy_(tensor_load)







def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2
    if args.normal_model:
        for net_i in range(n_parties):
            if args.model == 'simple-cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.use_project_head:
                net = ModelFedCon(args.model, args.out_dim, n_classes, net_configs)
            else:
                net = ModelFedCon_noheader(args.model, args.out_dim, n_classes, net_configs)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, device="cpu"):
    #net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().cuda()

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, _, out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch % 10 == 0:
            train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return train_acc, test_acc




def train_net_fedcon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                     args_optimizer, mu, temperature, args,
                     round, list_name, device="cpu"):
    #net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)




    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()
    # global_net.to(device)

    for previous_net in previous_nets:
        previous_net.cuda()
    #global_w = global_net.state_dict()

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)

            posi = cos(pro1, pro2)
            logits = posi.reshape(-1, 1)

            for previous_net in previous_nets:
                previous_net.cuda()
                _, pro3, _ = previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                previous_net.to('cpu')

            logits /= temperature
            labels = torch.zeros(x.size(0)).cuda().long()

            loss2 = mu * criterion(logits, labels)

            loss1 = criterion(out, target)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())



        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


        save_model_hos(net, list_name, net_id)

    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _ = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc




def main(aggregation_method,mu,seed,alg,epochs, dataset, model,communication_rounds,coef):
    args = get_args()

    args.mu=mu
    args.model=model
    args.dataset=dataset
    args.alg=alg
    args.epochs=epochs
    args.comm_round=communication_rounds


    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)





    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)



    print("len train_dl_global:", len(train_ds_global))


    train_dl = None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='cpu')

    global_model = global_models[0]



    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    if args.alg == 'moon':
        old_nets_pool = []
        if args.load_pool_file:
            for nets_id in range(args.model_buffer_size):
                old_nets, _, _ = init_nets(args.net_config, args.n_parties, args, device='cpu')
                checkpoint = torch.load(args.load_pool_file)
                for net_id, net in old_nets.items():
                    net.load_state_dict(checkpoint['pool' + str(nets_id) + '_' + 'net' + str(net_id)])
                old_nets_pool.append(old_nets)
        elif args.load_first_net:
            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False

        for round in range(n_comm_rounds):

            print('ROUND')
            print(round)

            logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_model.eval()
            for param in global_model.parameters():
                param.requires_grad = False
            global_w = global_model.state_dict()

            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            if round==0:
                list_name = getName(nets[0])


            for net in nets_this_round.values():



                net.load_state_dict(global_w)




            local_train_net(nets_this_round, args, net_dataidx_map, list_name, train_dl=train_dl, test_dl=test_dl,
                            global_model=global_model, prev_model_pool=old_nets_pool, round=round, device=device)

            #########

            print('CLIENTSSSS')
            print(args.n_parties)



            load_model_hos(global_model, list_name, args.n_parties, aggregation_method,coef)
            # summary(global_model.to(device), (3, 32, 32))

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl))
            global_model.cuda()
            train_acc, train_loss = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, device=device)

            print('global model acc')
            print(test_acc)

            global_model.to('cpu')
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)
            logger.info('>> Global Model Train loss: %f' % train_loss)

            if len(old_nets_pool) < args.model_buffer_size:
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_nets_pool.append(old_nets)
            elif args.pool_option == 'FIFO':
                old_nets = copy.deepcopy(nets)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(args.model_buffer_size - 2, -1, -1):
                    old_nets_pool[i] = old_nets_pool[i + 1]
                old_nets_pool[args.model_buffer_size - 1] = old_nets

            mkdirs(args.modeldir + 'fedcon/')
            if args.save_model:
                torch.save(global_model.state_dict(),
                           args.modeldir + 'fedcon/global_model_' + args.log_file_name + '.pth')
                torch.save(nets[0].state_dict(), args.modeldir + 'fedcon/localmodel0' + args.log_file_name + '.pth')
                for nets_id, old_nets in enumerate(old_nets_pool):
                    torch.save({'pool' + str(nets_id) + '_' + 'net' + str(net_id): net.state_dict() for net_id, net in
                                old_nets.items()},
                               args.modeldir + 'fedcon/prev_model_pool_' + args.log_file_name + '.pth')

def local_train_net(nets, args, net_dataidx_map,list_name, train_dl=None, test_dl=None, global_model=None, prev_model_pool=None,
                    server_c=None, clients_c=None, round=None, device="cpu"):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)



    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)



        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        prev_models = []
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])

        trainacc, testacc = train_net_fedcon(net_id, net, global_model, prev_models, train_dl_local, test_dl,
                                             n_epoch, args.lr,
                                             args.optimizer, args.mu, args.temperature, args, round,list_name, device=device)


        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        acc_list.append(testacc)

    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')
    return nets



if __name__ == '__main__':

    coef = 'k3k4'#can also be R0, Kl (R1), BH (R0.5), R2
    aggregation_method = 'cumulants_layer_prune'#aggregation method can also be the baseline average method (avg)

    dataset = 'cifar10' ##possible datasets: cifar10, cifar100, tinyimagenet
    model = 'simple-cnn'
    alg='moon'
    t_round=10
    communication_rounds = 100
    mu=8

    for i in range(3):
        main(aggregation_method, mu, i, alg, t_round, dataset, model, communication_rounds,coef)



    dataset='cifar100'
    model='resnet50-cifar100'
    communication_rounds=100
    mu=5


    for i in range(3):
        main(aggregation_method,  mu, i, alg, t_round, dataset, model, communication_rounds, coef)




    dataset = 'tinyimagenet'
    model = 'resnet50'
    mu=3
    communication_rounds = 20

    for i in range(3):
        main(aggregation_method, mu, i, alg, t_round, dataset, model, communication_rounds,  coef)





