from functools import partial
import torch 
from models.libs.logger import get_logger
from models.libs import utils
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np 
from .trainer import Trainer
torch.autograd.set_detect_anomaly(True)
class AdvTrainer(Trainer):
    def __init__(self,
                 data,
                 model_g,
                 model_d,
                 seq_length,
                 log_dir,
                 lr=0.0001,
                 max_epoch=100,
                 lr_scheduler=None,
                 optimizer_g=None,
                 optimizer_d=None,
                 max_grad_norm=5.0,
                 load_epoch=0,
                 max_t=100,
                 max_dt=10,
                 experiment_name='tpp_expriment',
                 device='cuda',
                 **kwargs):
        super().__init__(data, model_g, seq_length, log_dir, lr, max_epoch, 
                lr_scheduler, optimizer_g, max_grad_norm, load_epoch, max_t, max_dt, 
                experiment_name, device)

        self._model_g = model_g.to(device)
        self._model_d = model_d.to(device)   
        self.optimizer_g = torch.optim.Adam(self._model_g.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)\
            if optimizer_g is None else optimizer_g
        self.optimizer_d = torch.optim.Adam(self._model_d.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)\
            if optimizer_d is None else optimizer_d
        
        
    def train(self, log_every=1, g_times=5, test_every_n_epochs=10, save_model=True, save_every=True, patience=100):
        
        message = "the number of trainable parameters: " + str(utils.count_parameters(self._model))
        self._logger.info(message)
        self._logger.info('Start training the model ...')
        
        train_iterator = self.data['train_loader']
        
        # self._evaluate(dataset='test', metrics=['MAPE', 'CRPS', 'NLL', 'CE', 'LOG_LOSS', 'TOP1_ACC', 'TOP3_ACC']) 
        min_val_loss = float('inf')
        
        for epoch_num in range(self._epoch_num, self._max_epoch):

            self._model_g = self._model_g.train()
            self._model_d = self._model_d.train()

            epoch_train_loss = 0
            
            progress_bar = tqdm(train_iterator, unit="batch")
            for _, batch in enumerate(progress_bar):
                self.optimizer_g.zero_grad()
                self.optimizer_d.zero_grad()

                for i in range(g_times):
                    hist_embedding = self._model_g.learn(batch)
                    sample_t, _ = self._model_g.sample(batch, sample_num=1)
                    self.optimizer_g.zero_grad()
                    ce_loss = self._model_g.compute_ce(batch)
                    g_loss = self._model_d.g_loss(sample_t, batch, hist_embedding)
                    log_loss = g_loss + ce_loss
                    log_loss.backward(retain_graph=True)
                    self.optimizer_g.step()
                
                hist_embedding = self._model_g.learn(batch)
                sample_t, _ = self._model_g.sample(batch, sample_num=1)
                d_loss = self._model_d.d_loss(sample_t, batch, hist_embedding)
                d_loss.backward()
                self.optimizer_d.step()

                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.max_grad_norm)
                
                progress_bar.set_postfix(g_loss=g_loss.item(),
                                         d_loss=d_loss.item())
                
                epoch_train_loss += g_loss.detach()

            train_event_num = torch.sum(self._seq_lengths['train']).float()

            self.lr_scheduler.step()
            
            val_event_num, epoch_val_log_loss, epoch_val_nll_loss, epoch_val_ce_loss, epoch_val_mape, epoch_val_crps,\
                epoch_val_top1_acc, epoch_val_top3_acc, epoch_val_qqdev = self._evaluate(dataset='val')
            
            if (epoch_num % log_every) == log_every - 1:
                message = '---Epoch.{} Train Negative Overall Log-Loss per event: {:5f}. ' \
                    .format(epoch_num, epoch_train_loss / train_event_num)
                self._logger.info(message)

                message = '---Epoch.{} Val Negative Log-Loss per event: {:5f}; Negative Log-Likelihood per event:{:5f}; Cross-Entropy per event: {:5f}; MAPE: {:5f}; CRPS:{:5f}; Acc_Top1: {:5f}; Acc_Top3: {:5f}; QQ_dev: {:5f} ' \
                    .format(
                        epoch_num,
                        epoch_val_log_loss / val_event_num, 
                        epoch_val_nll_loss / val_event_num,
                        epoch_val_ce_loss / val_event_num, 
                        epoch_val_mape / val_event_num,
                        epoch_val_crps / val_event_num,
                        epoch_val_top1_acc / val_event_num,
                        epoch_val_top3_acc / val_event_num,
                        epoch_val_qqdev
                        )
                self._logger.info(message)
            
            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_event_num, epoch_test_log_loss, epoch_test_nll_loss, epoch_test_ce_loss, epoch_test_mape, epoch_test_crps,\
                    epoch_test_top1_acc, epoch_test_top3_acc, epoch_test_qqdev = self._evaluate(dataset='test')
                message = '---Epoch.{} Test Negative Log-Loss per event: {:5f}; Negative Log-Likelihood per event:{:5f}; Cross-Entropy per event: {:5f}; MAPE: {:5f}; CRPS:{:5f}; Acc_Top1: {:5f}; Acc_Top3: {:5f}; QQ_Dev: {:5f}' \
                    .format(
                        epoch_num,
                        epoch_test_log_loss / test_event_num, 
                        epoch_test_nll_loss / test_event_num,
                        epoch_test_ce_loss / test_event_num, 
                        epoch_test_mape / test_event_num,
                        epoch_test_crps / test_event_num,
                        epoch_test_top1_acc / test_event_num,
                        epoch_test_top3_acc / test_event_num,
                        epoch_test_qqdev
                        )
                self._logger.info(message)

            if save_every == True:
                model_file_name = self.save_model(epoch_num)
                self._logger.info(
                        'G-Loss is {:.5f}, '
                        'saving to {}'.format(epoch_val_log_loss / val_event_num, model_file_name))
            
            # find the best performance on validation set for epoch selection
            elif epoch_val_log_loss / val_event_num < min_val_loss:
                wait = 0
                if save_model:
                    best_epoch = epoch_num
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'G-Loss decrease from {:.5f} to {:.5f}, '
                        'saving to {}'.format(min_val_loss, epoch_val_log_loss / val_event_num, model_file_name))
                min_val_loss = epoch_val_log_loss / val_event_num
            # early stopping
            elif epoch_val_log_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num, 'the best epoch is: %d' % best_epoch)
                    break

    def _evaluate(self, dataset, epoch_num=0, load_model=False, metrics=['LOG_LOSS'], plot_qq=None):
        if load_model == True:
            self.load_model(epoch_num)
        self._model = self._model_g
        
        epoch_log_loss, epoch_nll_loss, epoch_ce_loss, epoch_mape, epoch_top1_acc, epoch_top3_acc, epoch_crps, epoch_qqdev\
            = 0, 0, 0, 0, 0, 0, 0, 0
        self._model.eval()
        val_iterator = self.data['{}_loader'.format(dataset)]
        
        report_metric = {}
        for metric in metrics:
            report_metric[metric] = 0 if metric != 'QQDEV' else []
            
        for _, batch in enumerate(val_iterator):
            self._model.evaluate(batch)
            
            for metric in metrics:
                value = self.eval_dict[metric](batch)
                if metric != 'QQDEV': 
                    report_metric[metric] += value.detach()
                    with torch.cuda.device('cuda:{}'.format(self._device.index)):
                        torch.cuda.empty_cache()
                    del value
                else:
                    report_metric[metric].append(value)
            
            self.optimizer.zero_grad()

        epoch_log_loss = report_metric['LOG_LOSS'] if 'LOG_LOSS' in report_metric else epoch_log_loss
        epoch_nll_loss = report_metric['NLL'] if 'NLL' in report_metric else epoch_nll_loss
        epoch_ce_loss = report_metric['CE'] if 'CE' in report_metric else epoch_ce_loss
        epoch_mape = report_metric['MAPE'] if 'MAPE' in report_metric else epoch_mape
        epoch_crps = report_metric['CRPS'] if 'CRPS' in report_metric else epoch_crps
        epoch_top1_acc = report_metric['TOP1_ACC'] if 'TOP1_ACC' in report_metric else epoch_top1_acc
        epoch_top3_acc = report_metric['TOP3_ACC'] if 'TOP3_ACC' in report_metric else epoch_top3_acc
            
        event_num = torch.sum(self._seq_lengths['{}'.format(dataset)]).float()

        if 'QQDEV' in metrics:
            cumrisk_samples = torch.cat(report_metric['QQDEV']).sort().values
            sample_num = cumrisk_samples.shape[0]
            probs = torch.linspace(0,1, 101)[1:-1].to(cumrisk_samples)
            estimate_quantiles = torch.le(cumrisk_samples[:,None], probs[None,:]).sum(dim=0)/sample_num
            exp1_quantiles = probs.log()
            epoch_qqdev = (estimate_quantiles - exp1_quantiles).abs().mean()
            if plot_qq is not None:
                import matplotlib.pyplot as plt
                plt.scatter(estimate_quantiles, exp1_quantiles)
                plt.savefig(plot_qq+'.png')

        event_num = torch.sum(self._seq_lengths['{}'.format(dataset)]).float()

        return event_num, epoch_log_loss, epoch_nll_loss, epoch_ce_loss, epoch_mape, epoch_crps, epoch_top1_acc, epoch_top3_acc, epoch_qqdev
          
    def final_test(self, n=1, metrics=['MAPE', 'CRPS', 'NLL', 'CE', 'LOG_LOSS', 'TOP1_ACC', 'TOP3_ACC', 'QQDEV']):
        model_path = Path(self._log_dir)/'saved_model'
        model_list = os.listdir(model_path)
        import re

        epoch_list = []
        for filename in model_list:
            epoch_list.append(int(re.search(r'\d+', filename).group()))

        epoch_list = np.sort(epoch_list)[-n:]
        loss_list = []
        for i in range(n):
            epoch_num = epoch_list[i]
            self.load_model(epoch_num)
            test_event_num, epoch_test_log_loss, epoch_test_nll_loss, epoch_test_ce_loss, epoch_test_mape, epoch_test_crps,\
                    epoch_test_top1_acc, epoch_test_top3_acc, epoch_test_qqdev = self._evaluate(dataset='test', metrics=metrics)
            message = '---Epoch.{} Test Negative Log-Loss per event: {:5f}; Negative Log-Likelihood per event:{:5f}; Cross-Entropy per event: {:5f}; MAPE: {:5f}; CRPS:{:5f}; Acc_Top1: {:5f}; Acc_Top3: {:5f}; QQ_Dev: {:5f}' \
                .format(
                    epoch_num,
                    epoch_test_log_loss / test_event_num, 
                    epoch_test_nll_loss / test_event_num,
                    epoch_test_ce_loss / test_event_num, 
                    epoch_test_mape / test_event_num,
                    epoch_test_crps / test_event_num,
                    epoch_test_top1_acc / test_event_num,
                    epoch_test_top3_acc / test_event_num,
                    epoch_test_qqdev
                )
            self._logger.info(message)
            loss_list.append((epoch_test_log_loss / test_event_num).item())
        test_loss_mean, test_loss_std = np.mean(loss_list), np.var(loss_list)
        message = 'Mean Negative_likelihood on test: {}, Variance: {}'.format(test_loss_mean, test_loss_std)
        self._logger.info(message)
    
    def compute_logloss(self, batch, sample_num=100):
        hist_embedding = self._model_g.learn(batch)
        sample_t, _ = self._model_g.sample(batch, sample_num=sample_num)
        seq_len, event_num = sample_t.shape[-2], sample_t.shape[-1]
        # sample_t = sample_t.reshape(-1,seq_len,event_num)
        return self._model_d.g_loss(sample_t, batch, hist_embedding)/sample_num
    
    def compute_nll(self, batch):
        return torch.tensor(0)
    
    def plot_similarity(self, file_name='type_similarity'):
        self._model_g.plot_similarity(file_name)
    
    
    