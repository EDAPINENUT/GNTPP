from functools import partial
import torch 
from models.libs.logger import get_logger
from models.libs import utils
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np 

class Trainer:
    def __init__(self,
                 data,
                 model,
                 seq_length,
                 log_dir,
                 lr=0.0001,
                 max_epoch=100,
                 lr_scheduler=None,
                 optimizer=None,
                 max_grad_norm=5.0,
                 load_epoch=0,
                 max_t=100,
                 max_dt=10,
                 experiment_name='tpp_expriment',
                 device='cuda',
                 **kwargs):
        
        self._device = device
        self.data = data
        self._seq_lengths = seq_length
        self._model = model.to(device)
        self._max_t = max_t.item() if torch.is_tensor(max_t) else (max_t)
        self._max_epoch = max_epoch
        self._max_dt = max_dt.item() if torch.is_tensor(max_dt) else (max_dt)
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)\
            if optimizer is None else optimizer
        
        steps = [20, 40, 60, 80]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=steps, gamma=0.5)\
            if lr_scheduler is None else lr_scheduler
        
        self.max_grad_norm = max_grad_norm
        
        self._experiment_name = experiment_name
        self._log_dir = self._get_log_dir(self, log_dir)
        self._logger = get_logger(self._log_dir, __name__, 'info.log', level='INFO')
        
        self._epoch_num = load_epoch
        if self._epoch_num > 0:
            self.load_model(self._epoch_num)
        
        self._init_evaluation_func()
        
    def _init_evaluation_func(self):
        self.eval_dict = {'MAPE': self.compute_mape,
                          'NLL': self.compute_nll,
                          'CE': self.compute_ce,
                          'LOG_LOSS': self.compute_logloss,
                          'TOP1_ACC': partial(self.compute_k_acc, k=1),
                          'TOP3_ACC': partial(self.compute_k_acc, k=3),
                          'CRPS': self.compute_crps,
                          'QQDEV': self.compute_qqplot_deviation}
    
    @staticmethod
    def _get_log_dir(self, log_dir):
        log_dir = Path(log_dir)/self._experiment_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        model_path = Path(self._log_dir)/'saved_model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        config = {}
        config['model_state_dict'] = self._model.state_dict()
        config['epoch'] = epoch
        model_name = model_path/('epo%d.tar' % epoch)
        torch.save(config, model_name)
        self._logger.info("Saved model at {}".format(epoch))
        return model_name
    
    def load_model(self, epoch_num):
        model_path = Path(self._log_dir)/'saved_model'
        model_name = model_path/('epo%d.tar' % epoch_num)

        assert os.path.exists(model_name), 'Weights at epoch %d not found' % epoch_num

        checkpoint = torch.load(model_name, map_location='cpu')
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch_num))
    
    def train(self, log_every=1, test_every_n_epochs=10, save_model=True, patience=100):
        
        message = "the number of trainable parameters: " + str(utils.count_parameters(self._model))
        self._logger.info(message)
        self._logger.info('Start training the model ...')
        
        train_iterator = self.data['train_loader']
        
        # self._evaluate(dataset='test', metrics=['QQDEV', 'CRPS', 'MAPE']) 
        min_val_loss = float('inf')
        
        for epoch_num in range(self._epoch_num, self._max_epoch):

            self._model = self._model.train()

            epoch_train_loss = 0
            
            progress_bar = tqdm(train_iterator, unit="batch")
            for _, batch in enumerate(progress_bar):
                
                self.optimizer.zero_grad()

                loss = self._model.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.max_grad_norm)
                
                progress_bar.set_postfix(training_loss=loss.item())
                self._logger.debug(loss.item())
                epoch_train_loss += loss.detach()
                

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

            # find the best performance on validation set for epoch selection
            if epoch_val_log_loss / val_event_num < min_val_loss:
                wait = 0
                if save_model:
                    best_epoch = epoch_num
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Negative Log-Loss decrease from {:.5f} to {:.5f}, '
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
        
        epoch_log_loss, epoch_nll_loss, epoch_ce_loss, epoch_mape, epoch_top1_acc, epoch_top3_acc, epoch_crps, epoch_qqdev\
            = 0, 0, 0, 0, 0, 0, 0, 0
        self._model.eval()
        val_iterator = self.data['{}_loader'.format(dataset)]
        
        report_metric = {}
        for metric in metrics:
            report_metric[metric] = 0 if metric != 'QQDEV' else []
        with torch.no_grad():
            for _, batch in enumerate(val_iterator):
                self._model.evaluate(batch)
                
                for metric in metrics:
                    value = self.eval_dict[metric](batch)
                    if metric != 'QQDEV': 
                        report_metric[metric] += value.detach()
                    else:
                        report_metric[metric].append(value)
                    with torch.cuda.device('cuda:{}'.format(self._device.index)):
                            torch.cuda.empty_cache()
                    del value
                
                self.optimizer.zero_grad()

        epoch_log_loss = report_metric['LOG_LOSS'] if 'LOG_LOSS' in report_metric else epoch_log_loss
        epoch_nll_loss = report_metric['NLL'] if 'NLL' in report_metric else epoch_nll_loss
        epoch_ce_loss = report_metric['CE'] if 'CE' in report_metric else epoch_ce_loss
        epoch_mape = report_metric['MAPE'] if 'MAPE' in report_metric else epoch_mape
        epoch_crps = report_metric['CRPS'] if 'CRPS' in report_metric else epoch_crps
        epoch_top1_acc = report_metric['TOP1_ACC'] if 'TOP1_ACC' in report_metric else epoch_top1_acc
        epoch_top3_acc = report_metric['TOP3_ACC'] if 'TOP3_ACC' in report_metric else epoch_top3_acc

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
    
    def compute_logloss(self, batch):
        return self._model.compute_loss(batch)
    
    def compute_mape(self, batch):
        out_dts, out_onehots = batch.out_dts, batch.out_onehots
        _max_dt = self._max_dt
        return self._mape_pred_time(self._model.predict_event_time(max_t=_max_dt), out_dts, out_onehots)
    
    def compute_nll(self, batch):
        return self._model.compute_nll(batch)
    
    def compute_ce(self, batch):
        return self._model.compute_ce(batch)
    
    def compute_k_acc(self, batch, k=1):
        return self._top_k_acc(self._model.predict_event_type(), batch.out_types, batch.out_onehots, top=k)

    def compute_crps(self, batch, sample_num=100):
        max_t = self._max_t
        out_dts, out_onehots = batch.out_dts, batch.out_onehots
        try:
            sample_dts, mask = self._model.sample(batch, sample_num=sample_num, max_t=max_t)
            sample_dts[sample_dts>=self._max_dt] = self._max_dt

        except:
            print('sampling process error!')
            return torch.tensor(0)
        
        sample_num = mask.sum(dim=1)
        
        nosample_event_num = ((mask.sum(dim=1) == 0) * out_onehots).sum()
        total_event_num = out_onehots.sum()
        reweights = total_event_num / (total_event_num - nosample_event_num)
        try:       
            diff_sample = ((sample_dts - out_dts[:,None,:,None]) * mask * out_onehots[:,None,...]).abs().sum(dim=1)/(sample_num)
        except:
            diff_sample = ((sample_dts - out_dts[:,None,:,None]) * mask * out_onehots[:,None,...].sum(dim=-1, keepdim=True)).abs().sum(dim=1)/(sample_num)
            
        diff_sample[diff_sample!=diff_sample] = 0.0
        diff_sample = diff_sample.sum()
        try:
            diff_distri = ((sample_dts[:,None,...] - sample_dts[:,:,None,...]) \
                * (mask[:,None,...] * mask[:,:,None,...]) * out_onehots[:,None,None,...]).abs().sum(dim=(1,2))/(sample_num**2 * 2)
        except:
            diff_distri = ((sample_dts[:,None,...] - sample_dts[:,:,None,...]) \
                * (mask[:,None,...] * mask[:,:,None,...]) * out_onehots[:,None,None,...].sum(dim=-1,keepdim=True)).abs().sum(dim=(1,2))/(sample_num**2 * 2)
        diff_distri[diff_distri!=diff_distri] = 0.0 # let Nan = 0
        diff_distri = diff_distri.sum()
        
        return (diff_sample - diff_distri)*reweights
    
    def _mape_pred_time(self, pred_time, batch_seq_dt, batch_one_hot):
            # relative absolute error
        try:
            if len(pred_time.shape) == 3:
                pred_time = pred_time
                per_event = torch.divide((pred_time.clamp(max=self._max_dt) - batch_seq_dt[:,:,None]), batch_seq_dt[:,:,None] + 1e-7)
                mask_event = (per_event.abs() * batch_one_hot).clamp(max=100.0)

            elif len(pred_time.shape) == 2:
                per_event = torch.divide((pred_time.clamp(max=self._max_dt) - batch_seq_dt), batch_seq_dt + 1e-7)
                mask_event = (per_event.abs()  * batch_one_hot.sum(dim=-1).bool()).clamp(max=100.0)
            return mask_event.sum()
        except:
            return torch.tensor(-1)

    def compute_qqplot_deviation(self, batch, sample_num=400, steps=40):
        cumulative_risk = self._model.cumulative_risk_func(batch, sample_num, steps=steps)
        one_hots = batch.out_onehots
        valid_cumrisk = cumulative_risk.expand_as(one_hots)[one_hots.bool()].flatten()
        return valid_cumrisk

    def _top_k_acc(self, pred_event_prob, batch_seq_type, batch_one_hot, top=5):
        
        # pred_event_prob: (batch_size, seq_len, event_num)
        # batch_seq_type: (batch_size, seq_len)
        try:
            top_pred = torch.argsort(pred_event_prob, dim=-1, descending=True)[...,:top]
            correct = top_pred.eq(batch_seq_type.unsqueeze(-1).expand_as(top_pred))
            correct_k = correct.view(-1).float().sum(0)
            return correct_k
        except:
            return torch.tensor(-1)
          
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
        
    def plot_similarity(self, file_name='type_similarity'):
        self._model.plot_similarity(file_name)