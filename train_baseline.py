import os
import time
import numpy as np
import torch
from opts.get_opts import Options
from data import create_dataset_with_args
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
                    fisher[n] += (1 - missing_rate[0]) * p.grad.pow(2).clone()
                elif 'netV' in n and 'attention_vector_weight' not in n:
                    fisher[n] += (1 - missing_rate[1]) * p.grad.pow(2).clone()
                elif 'netL' in n and 'attention_vector_weight' not in n:
                    fisher[n] += (1 - missing_rate[2]) * p.grad.pow(2).clone()
                elif 'netxenc_al2v' in n and 'attention_vector' not in n:
                    fisher[n] += (2 - missing_rate[0] - missing_rate[2]) / 2 * p.grad.pow(2).clone()
                elif 'netxenc_av2l' in n and 'attention_vector' not in n:
                    fisher[n] += (2 - missing_rate[0] - missing_rate[1]) / 2 * p.grad.pow(2).clone()
                elif 'netxenc_vl2a' in n and 'attention_vector' not in n:
                    fisher[n] += (2 - missing_rate[2] - missing_rate[1]) / 2 * p.grad.pow(2).clone()
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
        if n in fisher.keys():
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

    for i, data in enumerate(val_iter):  # inner loop within one epoch
        if mp is not None:
            data['missing_index'] = mp[opt.batch_size * i: opt.batch_size * (i + 1), :]
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()

        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        total_pred.append(pred)
        total_label.append(label)

    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)
    model.train()

    # save test results
    if is_save:
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)

    return acc, uar, f1, cm


def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join(opt.checkpoints_dir, expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))


if __name__ == '__main__':
    opt = Options().parse()  # get training options
    logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo))  # get logger path
    if not os.path.exists(logger_path):  # make sure logger path exists
        os.mkdir(logger_path)
    rate_list = [[0.5, 0.5, 0.5], [0.2, 0.5, 0.8], [0.8, 0.5, 0.2], [0, 0, 0]]
    missing_rate = rate_list[2]

    total_cv = 10 if opt.corpus_name == 'IEMOCAP' else 12
    result_recorder = ResultRecorder(os.path.join(opt.log_dir, opt.name, 'result.tsv'),
                                     total_cv=total_cv)  # init result recoreder
    suffix = '_'.join([opt.model, opt.dataset_mode])  # get logger suffix
    logger = get_logger(logger_path, suffix)  # get logger
    if opt.has_test:  # create a dataset given opt.dataset_mode and other options
        dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])
    else:
        dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])

    dataset_size = len(dataset)  # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # load model
    if os.path.exists('./checkpoints/fcw_fusion.pth'):
        model.load_state_dict(torch.load('./checkpoints/fcw_fusion.pth'))

    total_iters = 0  # the total number of training iterations
    best_eval_acc, best_eval_uar, best_eval_f1 = 0, 0, 0
    best_eval_epoch = -1  # record the best eval epoch
    mp = missing_pattern(3, dataset_size, missing_rate)
    # test ACC and forget rate
    if opt.first_test:
        _ = eval(model, val_dataset, is_save=True, phase='val', mp=mp)
        acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test')
        logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        logger.info('\n{}'.format(cm))
        result_recorder.write_result_to_tsv({
            'acc': acc,
            'uar': uar,
            'f1': f1
        }, cvNo=opt.cvNo)
    # check if fisher is existing.
    if not os.path.exists('./fisher/fisher.npy'):
        # without FCW
        for epoch in range(opt.epoch_count,
                           opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                total_iters += 1  # opt.batch_size
                epoch_iter += opt.batch_size

                data['missing_index'] = mp[opt.batch_size * i: opt.batch_size * (i + 1), :]
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

            # eval val set
            acc, uar, f1, cm = eval(model, val_dataset)
            logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                epoch, opt.niter + opt.niter_decay, acc, uar, f1))
            logger.info('\n{}'.format(cm))

            # show test result for debugging
            if opt.has_test and opt.verbose:
                acc, uar, f1, cm = eval(model, tst_dataset)
                logger.info('Tst result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                    epoch, opt.niter + opt.niter_decay, acc, uar, f1))
                logger.info('\n{}'.format(cm))

            # record epoch with best result
            if opt.corpus_name == 'IEMOCAP':
                if uar > best_eval_uar:
                    best_eval_epoch = epoch
                    best_eval_uar = uar
                    best_eval_acc = acc
                    best_eval_f1 = f1
                select_metric = 'uar'
                best_metric = best_eval_uar
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
    else:
        # use FCW
        fisher = np.load('./fisher/fisher.npy', allow_pickle=True).item()
        mean = np.load('./fisher/mean.npy', allow_pickle=True).item()
        for epoch in range(opt.epoch_count,
                           opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                total_iters += 1  # opt.batch_size
                epoch_iter += opt.batch_size
                data['missing_index'] = mp[opt.batch_size * i: opt.batch_size * (i + 1), :]
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.forward()
                # backward
                model.optimizer.zero_grad()
                loss_fcw = compute_fcw(model, fisher, mean)

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

            # eval val set
            acc, uar, f1, cm = eval(model, val_dataset)
            logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                epoch, opt.niter + opt.niter_decay, acc, uar, f1))
            logger.info('\n{}'.format(cm))

            # show test result for debugging
            if opt.has_test and opt.verbose:
                acc, uar, f1, cm = eval(model, tst_dataset)
                logger.info('Tst result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (
                    epoch, opt.niter + opt.niter_decay, acc, uar, f1))
                logger.info('\n{}'.format(cm))

            # record epoch with best result
            if opt.corpus_name == 'IEMOCAP':
                if uar > best_eval_uar:
                    best_eval_epoch = epoch
                    best_eval_uar = uar
                    best_eval_acc = acc
                    best_eval_f1 = f1
                select_metric = 'uar'
                best_metric = best_eval_uar
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
    # print best eval result
    logger.info('Best eval epoch %d found with %s %f' % (best_eval_epoch, select_metric, best_metric))
    torch.save(model.state_dict(), './checkpoints/fcw_fusion.pth')
    # save fisher
    fisher = None
    if os.path.exists('./fisher/fisher.npy'):
        fisher = np.load('./fisher/fisher.npy', allow_pickle=True).item()

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
    np.save('./fisher/mean.npy', mean)
    np.save('./fisher/fisher.npy', fisher)
    # test
    if opt.has_test:
        logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
        model.load_networks(best_eval_epoch)
        _ = eval(model, val_dataset, is_save=True, phase='val')
        acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test')
        logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
        logger.info('\n{}'.format(cm))
        result_recorder.write_result_to_tsv({
            'acc': acc,
            'uar': uar,
            'f1': f1
        }, cvNo=opt.cvNo)
    else:
        result_recorder.write_result_to_tsv({
            'acc': best_eval_acc,
            'uar': best_eval_uar,
            'f1': best_eval_f1
        }, cvNo=opt.cvNo)

    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)
