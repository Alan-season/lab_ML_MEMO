import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import copy
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import AdaptiveNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

num_workers=8

class MEMO(BaseLearner):
    """MEMO实现，继承自BaseLearner"""
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._old_base = None
        # 选择主干网络
        self._network = AdaptiveNet(args['convnet_type'], False)    # -> inc_net.py
        # train_base和train_adaptive是从命令行读取得到的bool值
        logging.info(f'>>> train generalized blocks:{self.args["train_base"]} train_adaptive:{self.args["train_adaptive"]}')

    def after_task(self):
        self._known_classes = self._total_classes
        if self._cur_task == 0:
            if self.args['train_base']:
                logging.info("Train Generalized Blocks...")
                self._network.TaskAgnosticExtractor.train()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = True
            else:
                logging.info("Fix Generalized Blocks...")
                self._network.TaskAgnosticExtractor.eval()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = False
        
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        # 每一个Task进行一次增量训练
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # 更新全连接层（初始化专用块）
        self._network.update_fc(self._total_classes)    # -> inc_net.py

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task>0:    # 如果不是第一次训练
            for i in range(self._cur_task):
                for p in self._network.AdaptiveExtractors[i].parameters():
                    if self.args['train_adaptive']:     # 如果是train_adaptive模式，则需要记录当前任务之前的专用块的梯度
                        p.requires_grad = True
                    else:   # 否则冻结之前专用块的梯度
                        p.requires_grad = False

        # 输出网络参数和可训练参数数量（因为可能有一些参数被冻结了）
        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        # 获取训练集
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),    # 在新的类上进行训练
            source='train',
            mode='train',
            # 这是baseLearner实现的一个方法，用于返回data和target的样例存储（初始是两个空数组）
            appendent=self._get_memory()    # -> base.py
        )   # -> data_manager.py
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args["batch_size"], 
            shuffle=True,
            num_workers=num_workers
        )
        
        # 获取测试集
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),  # 在所有的已知类上测试
            source='test', 
            mode='test'
        )   # -> data_manager.py
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args["batch_size"],
            shuffle=False, 
            num_workers=num_workers
        )

        # 并行设置
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        # 开始增量训练
        self._train(self.train_loader, self.test_loader)
        
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def set_network(self):
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.train()                   #All status from eval to train
        if self.args['train_base']:
            self._network.TaskAgnosticExtractor.train()
        else:
            self._network.TaskAgnosticExtractor.eval()
        
        # set adaptive extractor's status
        self._network.AdaptiveExtractors[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                if self.args['train_adaptive']:
                    self._network.AdaptiveExtractors[i].train()
                else:
                    self._network.AdaptiveExtractors[i].eval()
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            
    def _train(self, train_loader, test_loader):
        # 将数据搬到相应的设备上
        self._network.to(self._device)

        if self._cur_task==0:   # 初始训练阶段
            # 设置优化器
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"]
            )
            # 设置学习率调度器
            if self.args['scheduler'] == 'steplr':  # 指数衰减
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, 
                    milestones=self.args['init_milestones'], 
                    gamma=self.args['init_lr_decay']
                )
            elif self.args['scheduler'] == 'cosine':    # 余弦衰减
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['init_epoch']
                ) 
            else:
                raise NotImplementedError
            
            if not self.args['skip']:   # 进行初始训练
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            else:   # 读入已有模型
                if isinstance(self._network, nn.DataParallel):
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)

                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
                
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
        else:   # 增量学习阶段
            # 设置优化器
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                lr=self.args['lrate'], 
                momentum=0.9, 
                weight_decay=self.args['weight_decay']
            )
            # 设置学习率调度器
            if self.args['scheduler'] == 'steplr':  # 指数衰减
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=self.args['milestones'], 
                    gamma=self.args['lrate_decay']
                )
            elif self.args['scheduler'] == 'cosine':    # 余弦衰减
                assert self.args['t_max'] is not None
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['t_max']
                )
            else:
                raise NotImplementedError
            
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes-self._known_classes)
            else:
                self._network.weight_align(self._total_classes-self._known_classes)

            
    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        """初始训练阶段"""
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # inputs shape:[128, 3, 32, 32], targets shape:[128]
                logits = self._network(inputs)['logits']
                # logits shape:[128, 10]
                # 这里输出值的aux_logits等信息扔掉了，可能是初始阶段训练不需要这些
                # 计算交叉熵损失
                loss=F.cross_entropy(logits,targets.long()) # 这里需要把标签转换成long类型
                # 梯度归零
                optimizer.zero_grad()
                # 梯度反向传播
                loss.backward()
                # 更新网络参数
                optimizer.step()
                # 误差累积
                losses += loss.item()

                # 使用概率最大的类作为预测值
                _, preds = torch.max(logits, dim=1)
                # 统计正确值
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            # 动态调整学习率
            scheduler.step()
            # 计算训练集上的准确率
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            
            if epoch%5==0:
                # 每5代计算测试集上的准确率
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc)
            # prog_bar.set_description(info)
            logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """增量学习阶段"""
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.set_network()
            losses = 0.
            losses_clf=0.
            losses_aux=0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                outputs= self._network(inputs)
                logits,aux_logits=outputs["logits"],outputs["aux_logits"]
                loss_clf=F.cross_entropy(logits,targets.long())
                aux_targets = targets.clone()
                aux_targets=torch.where(aux_targets-self._known_classes+1>0,  aux_targets-self._known_classes+1,0)
                loss_aux=F.cross_entropy(aux_logits,aux_targets.long())
                loss=loss_clf+self.args['alpha_aux']*loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux+=loss_aux.item()
                losses_clf+=loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux  {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader),losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader), losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc)
            prog_bar.set_description(info)
        logging.info(info)