from VGG import vgg
import torch
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torchnet as tnt
import torch.nn.functional as F
from torch.autograd import Variable
import time
import os
import shutil
import random
import numpy as np
from scipy import stats
import scipy
import math
from scipy.special import softmax


from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets




def optimizer_to_cpu(optim):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.cpu()
            if param._grad is not None:
                param._grad.data = param._grad.data.cpu()
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.cpu()
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.cpu()


def getName(model):
    list_name = []

    for name in model.state_dict():
        list_name.append(name)

    return list_name


def seed_everything(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(SEED)


def save_checkpoint(state, is_best, counter, aggregation_method, weighted, n_clients, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath,
                                                                                   'model_best_test_acc_' + str(
                                                                                       counter) + '_' + str(
                                                                                       state['best_prec']) + '_' + str(
                                                                                       aggregation_method) + '_weighted=' + str(
                                                                                       weighted) +
                                                                                   '_n_clients=' + str(
                                                                                       n_clients) + '.pth.tar'))


def save_checkpoint_srv(state, is_best, counter, aggregation_method, weighted, checkpoint_prev, n_clients,  filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint_srv.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint_srv.pth.tar'),
                        os.path.join(filepath, 'model_best_test_acc_' + str(
                            counter) + '_' + str(state['best_prec']) + '_' + str(
                            aggregation_method) + '_weighted=' + str(
                            weighted) + '_n_clients=' + str(n_clients) + '.pth.tar'))
        if checkpoint_prev != 0:
            os.remove(checkpoint_prev)
        checkpoint_prev = os.path.join(filepath, 'model_best_test_acc_' + str(counter) + '_' + str(
            state['best_prec']) + '_' + str(aggregation_method) + '_weighted=' + str(weighted) + '_n_clients=' + str(
            n_clients) + '.pth.tar')

    return checkpoint_prev


def load_checkpoint_srv(filepath):
    if os.path.isfile(os.path.join(filepath, 'checkpoint_srv.pth.tar')):
        print("=> loading checkpoint '{}'".format(os.path.join(filepath, 'checkpoint_srv.pth.tar')))
        checkpoint = torch.load(os.path.join(filepath, 'checkpoint_srv.pth.tar'))
        print("=> loaded checkpoint '{}' ".format(os.path.join(filepath, 'checkpoint_srv.pth.tar')))
    else:
        print("=> no checkpoint found at '{}'".format(
            os.path.join(filepath, os.path.join(filepath, 'checkpoint_srv.pth.tar'))))
    return checkpoint


def load_checkpoint(best, counter, aggregation_method, weighted, n_clients, filepath):
    if os.path.isfile(os.path.join(filepath, 'model_best_test_acc_' + str(counter) + '_' + str(best) + '_' + str(
            aggregation_method) + '_weighted=' + str(weighted) +
                                             '_n_clients=' + str(n_clients) + '.pth.tar')):
        print("=> loading checkpoint '{}'".format(os.path.join(filepath,
                                                               'model_best_test_acc_' + str(counter) + '_' + str(
                                                                   best) + '_' + str(
                                                                   aggregation_method) + '_weighted=' + str(weighted) +
                                                               '_n_clients=' + str(n_clients) + '.pth.tar')))
        checkpoint = torch.load(os.path.join(filepath,
                                             'model_best_test_acc_' + str(counter) + '_' + str(best) + '_' + str(
                                                 aggregation_method) + '_weighted=' + str(weighted) +
                                             '_n_clients=' + str(n_clients) + '.pth.tar'))
        print("=> loaded checkpoint '{}'  Prec1: {:f}".format(os.path.join(filepath, 'model_best_test_acc_' + str(
            counter) + '_' + str(best) + '_' + str(aggregation_method) + '_weighted=' + str(weighted) +
                                                                           '_n_clients=' + str(n_clients) + '.pth.tar'),
                                                              best))
    else:
        print("=> no checkpoint found at '{}'".format(os.path.join(filepath,
                                                                   'model_best_test_acc_' + str(counter) + '_' + str(
                                                                       best) + '_' + str(
                                                                       aggregation_method) + '_weighted=' + str(
                                                                       weighted) +
                                                                   '_n_clients=' + str(n_clients) + '.pth.tar')))
    return checkpoint


def save_model(model, list_name, client):
    np_path = './SaveModel/Model_' + str(client) + '/'
    if os.path.isdir(np_path) == False:
        os.makedirs(np_path)

    for name in list_name:
        temp_num = model.state_dict()[name].cpu().numpy()
        np.save(np_path + "%s.ndim" % (name), temp_num)


def cumulants_filter_prune(temp_load_numpy, n_clients, temp_filter_cumulants,coef):
    temp = np.asarray(temp_load_numpy)

    if temp_load_numpy[0].ndim == 4 or temp_load_numpy[0].ndim == 2:


        if temp_load_numpy[0].ndim == 4:
            sz=temp_load_numpy[0].shape[1]*temp_load_numpy[0].shape[2]*temp_load_numpy[0].shape[3]
        else:
            sz = temp_load_numpy[0].shape[1]

        x=np.linspace(-1,1,sz)
        prob_g = stats.norm.pdf(x, 0, 1)

        ###
        temp_filter_cumulants = np.zeros((temp.shape[1], n_clients))
        ###

        temp_load_numpy_server = np.empty_like(temp_load_numpy[0])

        for i in range(np.shape(temp_load_numpy[0])[0]):

            cumulants_temp = []
            for client in range(n_clients):
                if temp_load_numpy[0].ndim == 4:

                    if coef == 'KL':

                        prob = softmax(temp_load_numpy[client][i, :, :, :].flatten())

                        cumulants_temp.append(stats.entropy(prob + 1e-10, prob_g + 1e-10))








                    elif coef == 'R0':

                        prob = softmax(temp_load_numpy[client][i, :, :, :].flatten())

                        if prob[i] != 0:
                            cumulants_temp.append(np.abs(math.log(sum([(prob_g[i]) for i in range(len(prob))]))))



                    elif coef == 'BH':

                        alpha = 1 / 2
                        prob = softmax(temp_load_numpy[client][i, :, :, :].flatten())

                        cumulants_temp.append(np.abs((1 / (alpha - 1)) * math.log(
                            sum([prob[i] ** alpha * prob_g[i] ** (1 - alpha) for i in range(len(prob))]))))

                    elif coef == 'R2':

                        alpha = 2
                        prob = softmax(temp_load_numpy[client][i, :, :, :].flatten())

                        cumulants_temp.append(np.abs((1 / (alpha - 1)) * math.log(
                            sum([prob[i] ** alpha * prob_g[i] ** (1 - alpha) for i in range(len(prob))]))))



                    elif coef=='k3k4':

                        cumulants_temp.append(abs(stats.kstat(temp_load_numpy[client][i, :, :, :], 4) * stats.kstat(
                            temp_load_numpy[client][i, :, :, :], 3)))



                else:
                    if coef == 'KL':

                        prob = softmax(temp_load_numpy[client][i, :].flatten())

                        cumulants_temp.append(stats.entropy(prob + 1e-10, prob_g + 1e-10))







                    elif coef == 'R0':

                        prob = softmax(temp_load_numpy[client][i, :].flatten())

                        if prob[i] != 0:
                            cumulants_temp.append(np.abs(math.log(sum([(prob_g[i]) for i in range(len(prob))]))))



                    elif coef == 'BH':

                        alpha = 1 / 2
                        prob = softmax(temp_load_numpy[client][i, :].flatten())

                        cumulants_temp.append(np.abs((1 / (alpha - 1)) * math.log(
                            sum([prob[i] ** alpha * prob_g[i] ** (1 - alpha) for i in range(len(prob))]))))

                    elif coef == 'R2':

                        alpha = 2
                        prob = softmax(temp_load_numpy[client][i, :].flatten())

                        cumulants_temp.append(np.abs((1 / (alpha - 1)) * math.log(
                            sum([prob[i] ** alpha * prob_g[i] ** (1 - alpha) for i in range(len(prob))]))))


                    elif coef== 'k3k4':

                        cumulants_temp.append(abs(stats.kstat(temp_load_numpy[client][i, :], 4) * stats.kstat(temp_load_numpy[client][i, :],
                                                                                        3)))




            temp_filter_cumulants[i, :] = cumulants_temp

            if temp_load_numpy[0].ndim == 4:

                temp_load_numpy_server[i, :, :, :] = np.average(temp[:, i, :, :, :], 0, temp_filter_cumulants[i, :])




            else:

                temp_load_numpy_server[i, :] = np.average(temp[:, i, :], 0, temp_filter_cumulants[i, :])





    elif temp_load_numpy[0].ndim == 1:

        temp_load_numpy_server = np.empty_like(temp_load_numpy[0])

        for i in range(np.shape(temp_load_numpy[0])[0]):
            temp_load_numpy_server[i] = np.average(temp[:, i], 0, temp_filter_cumulants[i, :])



    elif temp_load_numpy[0].ndim == 0:

        temp_load_numpy_server = np.average(temp, 0, np.average(temp_filter_cumulants, 0))

    return temp_load_numpy_server, temp_filter_cumulants


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

                prob = softmax(temp_load_numpy[client].flatten())



                t=[]
                for i in range(len(prob)):
                    if prob[i] != 0:
                        t.append(prob_g[i] )


                cumulants_temp.append(np.abs(math.log(sum(t))))





            elif coef=='BH':

                alpha=1/2
                prob = softmax(temp_load_numpy[client].flatten())


                cumulants_temp.append(np.abs( (1 / (alpha - 1)) * math.log(sum([prob[i] ** alpha * prob_g[i] ** (1 - alpha) for i in range(len(prob))]))))

            elif coef=='R2':

                alpha=2
                prob = softmax(temp_load_numpy[client].flatten())


                cumulants_temp.append(np.abs( (1 / (alpha - 1)) * math.log(sum([prob[i] ** alpha * prob_g[i] ** (1 - alpha) for i in range(len(prob))]))))


            elif coef=='k3k4':
                cumulants_temp.append(abs(stats.kstat(temp_load_numpy[client],4) * stats.kstat(temp_load_numpy[client], 3)))




        temp_filter_cumulants = cumulants_temp




        temp = np.asarray(temp_load_numpy)



        temp_load_numpy_server = np.average(temp, 0, temp_filter_cumulants)





    elif temp_load_numpy[0].ndim == 1 or temp_load_numpy[0].ndim == 0:

        temp = np.asarray(temp_load_numpy)
        temp_load_numpy_server = np.average(temp, 0, temp_filter_cumulants)

    return temp_load_numpy_server, temp_filter_cumulants


def cumulants_layer_prune_fl(temp_load_numpy, n_clients, temp_filter_cumulants):
    if temp_load_numpy[0].ndim == 4 or temp_load_numpy[0].ndim == 2:

        cumulants_temp = []
        for client in range(n_clients):
            cumulants_temp.append(abs(stats.kstat(temp_load_numpy[client],4) * stats.kstat(temp_load_numpy[client], 3)))

        temp_filter_cumulants = cumulants_temp

    elif temp_load_numpy[0].ndim == 1 or temp_load_numpy[0].ndim == 0:

        pass




    return  temp_filter_cumulants


def cumulants_filter_prune_fl(temp_load_numpy, n_clients, temp_filter_cumulants):
    temp = np.asarray(temp_load_numpy)

    if temp_load_numpy[0].ndim == 4 or temp_load_numpy[0].ndim == 2:

        ###
        temp_filter_cumulants = np.zeros((temp.shape[1], n_clients))
        ###

        for i in range(np.shape(temp_load_numpy[0])[0]):

            cumulants_temp = []
            for client in range(n_clients):
                if temp_load_numpy[0].ndim == 4:
                    cumulants_temp.append(abs(stats.kstat(temp_load_numpy[client][i, :, :, :], 4) * stats.kstat(
                        temp_load_numpy[client][i, :, :, :], 3)))
                else:
                    cumulants_temp.append(
                        abs(stats.kstat(temp_load_numpy[client][i, :], 4) * stats.kstat(temp_load_numpy[client][i, :],
                                                                                        3)))

            temp_filter_cumulants[i, :] = cumulants_temp











    elif temp_load_numpy[0].ndim == 1:

        pass



    elif temp_load_numpy[0].ndim == 0:

        pass

    return temp_filter_cumulants





def load_model(model, list_name, n_clients, aggregation_method, coef):
    np_path = './SaveModel/Model_'

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

        #print(var_name)

        for client in range(n_clients):
            temp_load_numpy.append(np.load(np_path + str(client) + "/%s.ndim.npy" % (var_name)))

        if aggregation_method == 'avg':

            temp = np.asarray(temp_load_numpy)
            temp_load_numpy_server = np.mean(temp, 0)


        elif aggregation_method == 'cumulants_layer_prune':

            temp_load_numpy_server, weight_avg = cumulants_layer_prune(temp_load_numpy, n_clients, weight_avg,coef)








        tensor_load = torch.tensor(temp_load_numpy_server)

        model.state_dict()[var_name].copy_(tensor_load)


def train(train_loader, epoch, model, optimizer):
    model.train()
    # avg_loss = 0.
    avg_loss = tnt.meter.AverageValueMeter()
    train_acc = 0.
    data_sum = 0

    for batch_idx, (data, target) in enumerate(train_loader):


        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss.add(loss.item())
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        output.detach().cpu()
        data_sum += data.size()[0]
        del output

        log_interval = 47
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss.item(), train_acc, data_sum,
                       100. * float(train_acc) / (float(data_sum))))

    return loss.item(), float(train_acc) / float(len(train_loader.sampler))


def test(model, test_loader):
    model.eval()
    test_loss = tnt.meter.AverageValueMeter()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = F.cross_entropy(output, target)
            test_loss.add(loss.item())  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            output.detach().cpu()
            del output

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            loss.item(), correct, len(test_loader.sampler),
            100. * float(correct) / len(test_loader.sampler)))
    return loss.item(), float(correct) / float(len(test_loader.sampler))


def main(aggregation_method, total_epochs, weighted, evaluate, path, splits, t_round, wd, normalization, n_clients,seed, coef,count,dataset):
    writer = SummaryWriter(log_dir='runs')

    seed_everything(seed)

    print('Local epochs are ' + str(t_round))

    checkpoint_prev = 0

    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomGrayscale(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if dataset=='cifar10':
        trainSet = torchvision.datasets.CIFAR10(root='./CifarTrainData', train=True,
                                            download=True, transform=transform)

    elif dataset=='cifar100':
        trainSet = torchvision.datasets.CIFAR100(root='./CifarTrainData', train=True,
                                                download=True, transform=transform)

    split_tr = []
    split_ev = []
    list_server = [int(len(trainSet) * 0.01)]
    total_len = len(trainSet) * 0.99

    for i in range(n_clients):
        if np.mod(i, 2) == 0:
            split_tr.append(int(np.floor(len(trainSet) * splits[i])))
            split_ev.append(int(np.floor(len(trainSet) * splits[n_clients + i])))
        else:
            split_tr.append(int(np.ceil(len(trainSet) * splits[i])))
            split_ev.append(int(np.ceil(len(trainSet) * splits[n_clients + i])))

    if sum(split_tr) > (total_len * 0.9):
        split_tr[-1] = int(split_tr[-1] - (sum(split_tr) - total_len * 0.9))
    elif sum(split_tr) < (total_len * 0.9):
        split_tr[-1] = int(split_tr[-1] - (sum(split_tr) - total_len * 0.9))

    if sum(split_ev) > (total_len * 0.1):
        split_ev[-1] = int(split_ev[-1] - (sum(split_ev) - total_len * 0.1))
    elif sum(split_ev) < (total_len * 0.1):
        split_ev[-1] = int(split_ev[-1] - (sum(split_ev) - total_len * 0.1))

    #print(sum(split_tr) + sum(split_ev) + sum(list_server))
    #print(len(trainSet))

    tot_dataset = torch.utils.data.random_split(trainSet,
                                                split_tr + split_ev + list_server,
                                                generator=torch.Generator().manual_seed(0))

    train_set = []
    valid_set = []
    valid_dataset_server = tot_dataset[-1]
    temp_dataset = tot_dataset[:-1]

    for i in range(n_clients):
        train_set.append(temp_dataset[i])
        valid_set.append(temp_dataset[n_clients + i])

    # n_clients=3
    batch_size = 64
    lr_next = np.zeros(n_clients)

    train_loader_list = []
    valid_loader_list = []
    optimizer_list = []

    for client in range(n_clients):
        train_loader_list.append(torch.utils.data.DataLoader(train_set[client], batch_size=batch_size, shuffle=True))
        valid_loader_list.append(torch.utils.data.DataLoader(valid_set[client], batch_size=batch_size, shuffle=False))

    valid_loader_server = torch.utils.data.DataLoader(valid_dataset_server, batch_size=batch_size, shuffle=False)

    if dataset=='cifar10':
        test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./CifarTrainData', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
                                                                                            (0.4914, 0.4822, 0.4465), (
                                                                                            0.2023, 0.1994, 0.2010))])))
    elif dataset=='cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./CifarTrainData', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
                                                                                                (0.4914, 0.4822, 0.4465),
                                                                                                (0.2023, 0.1994,0.2010))])))

    if evaluate == True:
        model = vgg(normalization, dataset, depth=16)
        model.cuda()

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])

        test(model, test_loader)

        return

    lr = 0.1
    momentum = 0.9
    weight_decay = wd
    save_path = './saved_models'

    try:
        os.makedirs(save_path)
        print("Directory '%s' created" % save_path)
    except FileExistsError:
        print("Directory '%s' already exists" % save_path)

    n_rounds = total_epochs // t_round

    best_prec = 0
    best_round = 0

    for round_n in range(n_rounds):

        for client in range(n_clients):

            print('Training of Client ' + str(client) + ' in Round ' + str(round_n))

            if round_n == 0:

                model = vgg(normalization, dataset, depth=16)
                model.cuda()




                if client == 0:
                    list_name = getName(model)

                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
                optimizer_list.append(optimizer)
                epoch_init = 0

            else:
                model = vgg(normalization, dataset, depth=16)
                model.cuda()

                epoch_init = round_n * t_round

                optimizer = optim.SGD(model.parameters(), lr=lr_next[client], momentum=momentum,
                                      weight_decay=weight_decay)
                checkpoint = load_checkpoint_srv(save_path)

                # best_prec[client] = checkpoint['best_prec']
                model.load_state_dict(checkpoint['state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer'])

            train_loader = train_loader_list[client]
            valid_loader = valid_loader_list[client]

            for epoch in range(epoch_init, epoch_init + t_round):

                if epoch in [int(total_epochs * 0.5), int(total_epochs * 0.75)]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.1

                print('LR')
                print(optimizer.param_groups[0]['lr'])
                lr_next[client] = optimizer.param_groups[0]['lr']

                start_time = time.time()
                train_loss, train_prec = train(train_loader, epoch, model, optimizer)
                valid_loss, valid_prec = test(model, valid_loader)

                writer.add_scalar("Loss/train" + str(client), train_loss, epoch)
                writer.add_scalar("Acc/train" + str(client), train_prec, epoch)
                writer.add_scalar("Loss/valid" + str(client), valid_loss, epoch)
                writer.add_scalar("Acc/valid" + str(client), valid_prec, epoch)

                writer.flush()

                
                save_model(model, list_name, client)
                elapsed_time = time.time() - start_time
                print('Elapsed time is ' + str(elapsed_time))

            model.cpu()
            optimizer_to_cpu(optimizer)
            writer.flush()

        model = vgg(normalization, dataset, depth=16)
        model.cuda()

        load_model(model, list_name, n_clients, aggregation_method, coef)

        loss_server, prec_server = test(model, valid_loader_server)

        if prec_server > best_prec:
            is_best = True
            best_prec = prec_server
            best_round = round_n + 1
        else:
            is_best = False

        writer.add_scalar("Loss/valid_server", loss_server, round_n)
        writer.add_scalar("Acc/valid_server", prec_server, round_n)

        checkpoint_prev = save_checkpoint_srv({
            'epoch': round_n + 1,
            'state_dict': model.state_dict(),
            'best_prec': prec_server,
            'optimizer': 0,
        }, is_best, str(count)+'_'+str(round_n + 1), aggregation_method, weighted, checkpoint_prev, n_clients, filepath=save_path)

        print('The precision of the server is ' + str(prec_server))

        model.cpu()
        writer.flush()

    writer.close()

    model.cuda()

    checkpoint = load_checkpoint(best_prec, str(count)+'_'+str(best_round), aggregation_method, weighted, n_clients, save_path)

    model.load_state_dict(checkpoint['state_dict'])

    loss_server, prec_server = test(model, test_loader)

    save_checkpoint({
        'epoch': round_n + 1,
        'state_dict': model.state_dict(),
        'best_prec': prec_server,
        'optimizer': 0,
    }, True, 'test_result_'+str(count), aggregation_method, weighted, n_clients, filepath=save_path)

    print('The precision of the server at test set is ' + str(prec_server))

    return prec_server


if __name__ == '__main__':

    path = ''


    weighted = True

    wd_factor = 1e-4
    dataset='cifar100'   ##possible datasets: cifar10, cifar100

    normalization = 'normal'


    t_round=50
    clients=3
    count=0



    epochs=300



    coef = 'k3k4' #can also be R0, Kl (R1), BH (R0.5), R2

    for j in range(1):

        splits = []
        for i in range(clients):
            splits.append(0.891 / clients)

        for i in range(clients):
            splits.append(0.099 / clients)

        aggregation_method = 'cumulants_filter_prune' #aggregation method can also be the baseline average method


        main(aggregation_method, epochs, weighted, False, path, splits, t_round,wd_factor,normalization,clients,j,coef,count,dataset)
        count+=1




























































