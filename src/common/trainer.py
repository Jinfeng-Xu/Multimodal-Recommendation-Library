# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk

import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator
from utils.visualization import TrainingVisualizer


class AbstractTrainer(object):
    """Base trainer class"""
    def __init__(self, config, model):
        self.config = config
        self.model = model
    
    def fit(self, train_data):
        raise NotImplementedError()

    def evaluate(self, eval_data):
        raise NotImplementedError()


class Trainer(AbstractTrainer):
    """Basic trainer - Mirror Gradient and MDVT removed"""
    
    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config.get('clip_grad_norm', None)
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        
        # Weight decay
        self.weight_decay = 0.0
        if config.get('weight_decay') is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config.get('req_training', True)
        self.start_epoch = 0
        self.cur_step = 0

        # Initialize best results
        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        
        # Optimizer
        self.optimizer = self._build_optimizer()

        # Learning rate scheduler
        lr_scheduler = config.get('learning_rate_scheduler', [1.0, 50])
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        # Evaluator
        self.eval_type = config.get('eval_type', 'full')
        self.evaluator = TopKEvaluator(config)
        
        # Visualizer (saved to log/{model}/{dataset}/visualization/)
        self.visualizer = TrainingVisualizer(config)

    def _build_optimizer(self):
        """Build optimizer"""
        if self.learner.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            return optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning(f'Unknown optimizer: {self.learner}, using Adam')
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        """Train for one epoch"""
        if not self.req_training:
            return 0.0, []
            
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            
            # Compute loss
            losses = loss_func(interaction)
            
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            
            # Check for NaN
            if torch.isnan(loss):
                self.logger.info(f'Loss is nan at epoch: {epoch_idx}, batch index: {batch_idx}')
                return loss, torch.tensor(0.0)

            # Backward propagation (Mirror Gradient logic removed)
            loss.backward()
            
            # Gradient clipping
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            
            self.optimizer.step()
            loss_batches.append(loss.detach())

        return total_loss, loss_batches

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        """Validate for one epoch"""
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        """Check for NaN"""
        return torch.isnan(loss)

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        """Generate training loss output"""
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        """Train the model"""
        for epoch_idx in range(self.start_epoch, self.epochs):
            # Training
            training_start_time = time()
            if hasattr(self.model, 'pre_epoch_processing'):
                self.model.pre_epoch_processing()
            
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            
            if torch.is_tensor(train_loss):
                break  # NaN loss
            
            self.lr_scheduler.step()
            train_loss_value = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            self.train_loss_dict[epoch_idx] = train_loss_value
            
            # Record training loss for visualization
            self.visualizer.record_loss(epoch_idx, train_loss_value)
            
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            
            if verbose:
                # Beautify output format (enlarged by 1.3x)
                box_width = 92  # 70 * 1.3 ≈ 92
                time_str = f"{training_end_time - training_start_time:.2f}s"
                self.logger.info(f"╔{'═' * box_width}╗")
                self.logger.info(f"║ Epoch {epoch_idx + 1:3d}/{self.epochs:3d} │ Train Loss: {train_loss_value:.4f} │ Time: {time_str}")
                self.logger.info(f"╚{'═' * box_width}╝")
                if hasattr(self.model, 'post_epoch_processing'):
                    post_info = self.model.post_epoch_processing()
                    if post_info is not None:
                        self.logger.info(post_info)

            # Evaluation
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                
                # Test
                _, test_result = self._valid_epoch(test_data)
                
                if verbose:
                    # Beautify evaluation output (enlarged by 1.3x)
                    box_width = 92  # 70 * 1.3 ≈ 92
                    time_str = f"{valid_end_time - valid_start_time:.2f}s"
                    self.logger.info(f"╔{'═' * box_width}╗")
                    self.logger.info(f"║ Evaluation │ Valid: {valid_score:.4f} │ Time: {time_str}")
                    self.logger.info(f"║ Metrics: {dict2str(valid_result)}")
                    self.logger.info(f"║ Test: {dict2str(test_result)}")
                    self.logger.info(f"╚{'═' * box_width}╝")
                
                # Record metrics for visualization (auto-update best results)
                self.visualizer.record_metrics(epoch_idx, valid_result, test_result)
                
                if update_flag:
                    update_output = f"🏆 {self.config['model']} - Best {self.config['valid_metric']} = {valid_score:.4f} (Epoch {epoch_idx + 1})"
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                if stop_flag:
                    box_width = 92
                    stop_output = f"╔{'═' * box_width}╗\n"
                    stop_output += f"║ ✅ Training finished at epoch {epoch_idx + 1} | Best at epoch {self.visualizer.best_epoch + 1}\n"
                    stop_output += f"║ 🏆 {self.config['valid_metric']} = {self.visualizer.best_valid_score:.4f}\n"
                    stop_output += f"╚{'═' * box_width}╝"
                    if verbose:
                        self.logger.info(stop_output)
                    break
        
        # Generate visualization charts for best results
        self.visualizer.plot_all()
        
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid

    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        """Evaluate the model"""
        self.model.eval()
        batch_matrix_list = []
        
        for batch_idx, batched_data in enumerate(eval_data):
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            
            # Mask positive samples
            scores[masked_items[0], masked_items[1]] = -1e10
            
            # Get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)
            batch_matrix_list.append(topk_index)
        
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        """Plot training loss curve"""
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)