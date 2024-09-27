import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from codecarbon import EmissionsTracker

EPSILON = 1e-8
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args, outpath):
        super().__init__(args, outpath)
        self._network = IncrementalNet(args, True)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        task_emission_tracker = EmissionsTracker(log_level="critical", project_name="ICARL_Task_{}".format(self._cur_task), output_file=self.outpath+"ICARL_per_task_emissions.csv")
        task_emission_tracker.start()
        self._train(self.train_loader, self.test_loader)
        task_emission_tracker.stop()
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                [
                      {'params':self._network.backbone.parameters()},
                      {'params':self._network.fc.parameters(),'lr':0.01}
                ],
                momentum=0.9,
                lr=0.0001,
                weight_decay=0.0005,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=[1000], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                [
                      {'params':self._network.backbone.parameters()},
                      {'params':self._network.fc.parameters(),'lr':0.01}
                ],
                momentum=0.9,
                lr=0.0001,
                weight_decay=0.0005,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=[1000], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            epoch_emission_tracker = EmissionsTracker(log_level="critical", project_name="ICARL_Task_{}_Epoch_{}".format(self._cur_task,epoch), output_file=self.outpath+"ICARL_per_epoch_emissions.csv")
            epoch_emission_tracker.start()
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.args["only_inference"] == "y":
                    break
            epoch_emission_tracker.stop()
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            epoch_emission_tracker = EmissionsTracker(log_level="critical", project_name="ICARL_Task_{}_Epoch_{}".format(self._cur_task,epoch), output_file=self.outpath+"ICARL_per_epoch_emissions.csv")
            epoch_emission_tracker.start()
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    self.args["T"],
                )

                loss = loss_clf + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                if self.args["only_inference"] == "y":
                    break
            epoch_emission_tracker.stop()
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)


    def inference_gpu_time(self, model_name, outpath):
        self._network.to(self._device)
        self._network.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(self._device)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 10000
        timings=np.zeros((repetitions,1))

        #GPU-WARM-UP
        for _ in range(100):
            _ = self._network(dummy_input)

        print("Measuring the inference time on GPU for {}...".format(model_name))
        with torch.no_grad():
            inference_tracker = EmissionsTracker(log_level="critical", project_name="ICARL_inference_Task_{}".format(self._cur_task), output_file=self.outpath+"ICARL_per_task_inference_emissions.csv")
            inference_tracker.start()
            for rep in range(repetitions):
                starter.record()
                self._network(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
            inference_tracker.stop()
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print("GPU Mean: {}. GPU Std: {}".format(mean_syn, std_syn))
        np.save(outpath+"{}_gpu_inference_time.npy".format(model_name), timings)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
