import os
import time
import numpy as np
import torch

from opts.get_opts import Options
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from missing_index import missing_pattern

lrate = 0.0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getFisherDiagonal(opt, model, train_loader, missing_rate, mp=None):
    fisher = {
        n: torch.zeros(p.shape).cuda()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    model.train()
    for i, data in enumerate(train_loader):
        if mp is not None:
            data['missing_index'] = mp[opt.batch_size * i: opt.batch_size * (i + 1),
                                    :]  # A=missing[:,0] V=missing[:,1]  L=missing[:,2]
        model.set_input(data)
        model.forward()
        # backward
        model.optimizer.zero_grad()
        model.backward()
        model.optimizer.step()
        for n, p in model.named_parameters():
            if p.grad is not None:
                # Calculate the weights of different modality
                if 'netA' in n and 'attention_vector_weight' not in n:
                    fisher[n] += (1-missing_rate[0]) * p.grad.pow(2).clone()
                elif 'netV' in n and 'attention_vector_weight' not in n:
                    fisher[n] += (1-missing_rate[1]) * p.grad.pow(2).clone()
                elif 'netL' in n and 'attention_vector_weight' not in n:
                    fisher[n] += (1-missing_rate[2]) * p.grad.pow(2).clone()
                elif 'netxenc_al2v' in n and 'attention_vector' not in n:
                    fisher[n] += (2-missing_rate[0] - missing_rate[2]) / 2 * p.grad.pow(2).clone()
                elif 'netxenc_av2l' in n and 'attention_vector' not in n:
                    fisher[n] += (2-missing_rate[0] - missing_rate[1]) / 2 * p.grad.pow(2).clone()
                elif 'netxenc_vl2a' in n and 'attention_vector' not in n:
                    fisher[n] += (2-missing_rate[2] - missing_rate[1]) / 2 * p.grad.pow(2).clone()
                else:
                    fisher[n] += p.grad.pow(2).clone()
                # fisher[n] += p.grad.pow(2).clone()
    for n, p in fisher.items():
        fisher[n] = p / len(train_loader)
    return fisher


def to_gpu(data):
    if isinstance(data, dict):
        return {k: to_gpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_gpu(i) for i in data]
    elif isinstance(data, tuple):
        return tuple([to_gpu(i) for i in data])
    elif torch.is_tensor(data) and data.device != device:
        return data.to(device)
    else:
        return data


def compute_fcw(model, fisher, mean):
    loss = 0
    for n, p in model.named_parameters():
        if n in fisher.keys() and 'attention_vector_weight' not in n:
            loss += (
                    torch.sum(
                        (fisher[n])
                        * (p[: len(mean[n])] - mean[n]).pow(2)
                    )
                    / 2
            )
    return loss


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval(model, val_iter, is_save=False, phase='test', mp=None):
    model.eval()
    total_pred = []
    total_label = []
    total_miss_type = []
    for _, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        miss_type = np.array(data['miss_type'])
        total_pred.append(pred)
        total_label.append(label)
        total_miss_type.append(miss_type)

    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    total_miss_type = np.concatenate(total_miss_type)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)

    if is_save:
        # save test whole results
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

        # save part results
        for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl', 'nnn']:
            part_index = np.where(total_miss_type == part_name)
            part_pred = total_pred[part_index]
            part_label = total_label[part_index]
            acc_part = accuracy_score(part_label, part_pred)
            uar_part = recall_score(part_label, part_pred, average='macro')
            f1_part = f1_score(part_label, part_pred, average='macro')
            np.save(os.path.join(save_dir, '{}_{}_pred.npy'.format(phase, part_name)), part_pred)
            np.save(os.path.join(save_dir, '{}_{}_label.npy'.format(phase, part_name)), part_label)
            if phase == 'test':
                recorder_lookup[part_name].write_result_to_tsv({
                    'acc': acc_part,
                    'uar': uar_part,
                    'f1': f1_part
                }, cvNo=opt.cvNo)

    model.train()

    return acc, uar, f1, cm


def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join(opt.checkpoints_dir, expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))


if __name__ == '__main__':
    opt = Options().parse()  # get training options
    opt.ext = 1.5

    opt.mse_weight = 0.15
    opt.ii = 3
    opt.beta = 0.7
    opt.eta = 0.1

    rate_list = [[0.5, 0.5, 0.5], [0.2, 0.5, 0.8], [0.8, 0.5, 0.2], [0, 0, 0]]
    missing_rate = rate_list[2]
    print(missing_rate)
    # missing_rate = [0.5, 0.5, 0.5]
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo))  # get logger path
    if not os.path.exists(logger_path):  # make sure logger path exists
        os.mkdir(logger_path)

    result_dir = os.path.join(opt.log_dir, opt.name, 'results')
    if not os.path.exists(result_dir):  # make sure result path exists
        os.mkdir(result_dir)

    total_cv = 10 if opt.corpus_name == 'IEMOCAP' else 12
    recorder_lookup = {  # init result recoreder
        "total": ResultRecorder(os.path.join(result_dir, 'result_total.tsv'), total_cv=total_cv),
        "azz": ResultRecorder(os.path.join(result_dir, 'result_azz.tsv'), total_cv=total_cv),
        "zvz": ResultRecorder(os.path.join(result_dir, 'result_zvz.tsv'), total_cv=total_cv),
        "zzl": ResultRecorder(os.path.join(result_dir, 'result_zzl.tsv'), total_cv=total_cv),
        "avz": ResultRecorder(os.path.join(result_dir, 'result_avz.tsv'), total_cv=total_cv),
        "azl": ResultRecorder(os.path.join(result_dir, 'result_azl.tsv'), total_cv=total_cv),
        "zvl": ResultRecorder(os.path.join(result_dir, 'result_zvl.tsv'), total_cv=total_cv),
        "nnn": ResultRecorder(os.path.join(result_dir, 'result_avl.tsv'), total_cv=total_cv),
    }

    suffix = '_'.join([opt.model, opt.dataset_mode])  # get logger suffix
    logger = get_logger(logger_path, suffix)  # get logger

    if opt.has_test:  # create a dataset given opt.dataset_mode and other options
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])
    dataset_size = len(dataset)  # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
    opt.total_iters = (opt.niter + opt.niter_decay + 1) * dataset_size
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # load model
    if os.path.exists('./checkpoints/FCW_miss.pth'):
        model.load_state_dict(torch.load('./checkpoints/FCW_miss.pth'))

    total_iters = 0  # the total number of training iterations
    best_eval_epoch = -1  # record the best eval epoch
    best_eval_acc, best_eval_uar, best_eval_f1 = 0, 0, 0

    mp = missing_pattern(3, dataset_size, missing_rate)
    # mp = missing_pattern(3, dataset_size, [0.5, 0.5, 0.5])    ##################

    perf = []
    # test ACC and forget rate
    if opt.first_test:
        _ = eval(model, val_dataset, is_save=True, phase='val')
        acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test')
        logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        logger.info('\n{}'.format(cm))
        recorder_lookup['total'].write_result_to_tsv({
            'acc': acc,
            'uar': uar,
            'f1': f1
        }, cvNo=opt.cvNo)
    # check if fisher is existing.
    if not os.path.exists('./fisher/fisher_miss.npy'):
        fisher = None
        mean = None
        for epoch in range(opt.epoch_count,
                           opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            print("NO FCW")
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                total_iters += 1  # opt.batch_size
                epoch_iter += opt.batch_size
                data['missing_index'] = mp[opt.batch_size * i: opt.batch_size * (i + 1), :]  #################

                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters(epoch)  # calculate loss functions, get gradients, update network weights

                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    logger.info('Cur epoch {}'.format(epoch) + ' loss ' +
                                ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(
                                    **losses))

                iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (
                epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate(logger)  # update learning rates at the end of every epoch.

            # eval
            acc, uar, f1, cm = eval(model, val_dataset)
            logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                epoch, opt.niter + opt.niter_decay, acc, uar, f1))
            logger.info('\n{}'.format(cm))

            # show test result for debugging
            if opt.has_test and opt.verbose:
                acc, uar, f1, cm = eval(model, tst_dataset)
                logger.info('Tst result of epoch %d acc %.4f uar %.4f f1 %.4f' % (epoch, acc, uar, f1))
                logger.info('\n{}'.format(cm))

            perf.append([acc, uar, f1])

            # record epoch with best result
            if opt.corpus_name == 'IEMOCAP':
                if f1 > best_eval_f1:
                    best_eval_epoch = epoch
                    best_eval_uar = uar
                    best_eval_acc = acc
                    best_eval_f1 = f1
                select_metric = 'f1'
                best_metric = best_eval_f1
            elif opt.corpus_name == 'MSP':
                if f1 > best_eval_f1:
                    best_eval_epoch = epoch
                    best_eval_uar = uar
                    best_eval_acc = acc
                    best_eval_f1 = f1
                select_metric = 'f1'
                best_metric = best_eval_f1
            else:

                raise ValueError(f'corpus name must be IEMOCAP or MSP, but got {opt.corpus_name}')
        # save fisher
        if fisher is None:
            fisher = getFisherDiagonal(opt, model, dataset, missing_rate, mp)
            fisher = to_gpu(fisher)
        else:
            alpha = 1
            new_fisher = getFisherDiagonal(opt, model, dataset, missing_rate, mp)
            new_fisher = to_gpu(new_fisher)
            for n, p in new_fisher.items():
                new_fisher[n][: len(fisher[n])] = (
                        alpha * fisher[n]
                        + (1 - alpha) * new_fisher[n][: len(fisher[n])]
                )
            fisher = new_fisher
        mean = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        print(compute_fcw(model, fisher, mean))
        np.save('./fisher/mean_miss.npy', mean)
        np.save('./fisher/fisher_miss.npy', fisher)

    else:
        fisher = np.load('./fisher/fisher_miss.npy', allow_pickle=True).item()
        mean = np.load('./fisher/mean_miss.npy', allow_pickle=True).item()
        print(compute_fcw(model, fisher, mean))
        for epoch in range(opt.epoch_count,
                           opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            print("FCW!!!!!!!")
            print(compute_fcw(model, fisher, mean))
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                total_iters += 1  # opt.batch_size
                epoch_iter += opt.batch_size
                data['missing_index'] = mp[opt.batch_size * i: opt.batch_size * (i + 1), :]
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                loss_fcw = compute_fcw(model, fisher, mean)
                # forward
                model.forward()
                # backward
                model.optimizer.zero_grad()
                model.backward_fcw(loss_fcw)
                model.optimizer.step()

                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    logger.info('Cur epoch {}'.format(epoch) + ' loss ' +
                                ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(
                                    **losses))

                    iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (
                epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate(logger)  # update learning rates at the end of every epoch.

            # eval
            acc, uar, f1, cm = eval(model, val_dataset)
            logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                epoch, opt.niter + opt.niter_decay, acc, uar, f1))
            logger.info('\n{}'.format(cm))

            # show test result for debugging
            if opt.has_test and opt.verbose:
                acc, uar, f1, cm = eval(model, tst_dataset)
                logger.info('Tst result of epoch %d acc %.4f uar %.4f f1 %.4f' % (epoch, acc, uar, f1))
                logger.info('\n{}'.format(cm))

            perf.append([acc, uar, f1])

            # record epoch with best result
            if opt.corpus_name == 'IEMOCAP':
                if f1 > best_eval_f1:
                    best_eval_epoch = epoch
                    best_eval_uar = uar
                    best_eval_acc = acc
                    best_eval_f1 = f1
                select_metric = 'f1'
                best_metric = best_eval_f1
            elif opt.corpus_name == 'MSP':
                if f1 > best_eval_f1:
                    best_eval_epoch = epoch
                    best_eval_uar = uar
                    best_eval_acc = acc
                    best_eval_f1 = f1
                select_metric = 'f1'
                best_metric = best_eval_f1
            else:
                raise ValueError(f'corpus name must be IEMOCAP or MSP, but got {opt.corpus_name}')
        if fisher is None:
            fisher = getFisherDiagonal(opt, model, dataset, missing_rate, mp)
            fisher = to_gpu(fisher)
        else:
            alpha = 1
            new_fisher = getFisherDiagonal(opt, model, dataset, missing_rate, mp)
            new_fisher = to_gpu(new_fisher)
            for n, p in new_fisher.items():
                new_fisher[n][: len(fisher[n])] = (
                        alpha * fisher[n]
                        + (1 - alpha) * new_fisher[n][: len(fisher[n])]
                )
            fisher = new_fisher
        mean = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        print(compute_fcw(model, fisher, mean))
        np.save('./fisher/mean_miss.npy', mean)
        np.save('./fisher/fisher_miss.npy', fisher)
    logger.info('Best eval epoch %d found with %s %f' % (best_eval_epoch, select_metric, best_metric))

    # test
    if opt.has_test:
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)
        torch.save(model.state_dict(), './checkpoints/fcw_miss.pth')
        _ = eval(model, val_dataset, is_save=True, phase='val')
        acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test')
        logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        logger.info('\n{}'.format(cm))
        recorder_lookup['total'].write_result_to_tsv({
            'acc': acc,
            'uar': uar,
            'f1': f1
        }, cvNo=opt.cvNo)
    else:
        recorder_lookup['total'].write_result_to_tsv({
            'acc': best_eval_acc,
            'uar': best_eval_uar,
            'f1': best_eval_f1
        }, cvNo=opt.cvNo)

    # np.save('./fisher/mean_miss.npy', mean)
    # np.save('./fisher/fisher_miss.npy', fisher)
    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)
    logger.info('The number of training samples = %d' % dataset_size)
