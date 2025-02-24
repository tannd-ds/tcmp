import os
import shutil

import yaml
from torch import nn
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset import TrackingDataset, custom_collate_fn, augment_data
from utils import calculate_iou, calculate_ade, original_shape
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

from tcmp.models.tcn_new import DilatedCausalConvNet
from tcmp.tracking_utils.visualization import plot_tracking


class Tracker(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 0

        self._init_model_dir()
        self._init_model()
        if not self.config['eval']:
            self._init_writer()
            self._init_data_loader()
            self._init_optimizer()

            os.makedirs(os.path.join(self.config['model_dir'], 'weights'), exist_ok=True)


    def train(self):
        """ Train the model """
        torch.backends.cudnn.benchmark = True
        k_fold = KFold(n_splits=5, shuffle=True)

        print("Training the model...")
        for epoch in range(self.config['epochs']):
            train_indices, val_indices = next(k_fold.split(self.train_data))
            train_set = torch.utils.data.Subset(self.train_data, train_indices)
            val_set = torch.utils.data.Subset(self.train_data, val_indices)

            train_set_loader = DataLoader(
                train_set,
                batch_size=self.config['batch_size'],
                shuffle=True,
                collate_fn=custom_collate_fn,
                num_workers=4,
                pin_memory=True,
            )
            val_set_loader = DataLoader(
                val_set,
                batch_size=self.config['batch_size'],
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=4,
                pin_memory=True,
            )

            print('\033[92m', end='') # green
            self.step(train_set_loader, train=True)
            print('\033[93m', end='') # yellow
            self.step(val_set_loader, train=False, log_writer=False)
            print('\033[0m', end='') # white

            if (epoch + 1) % self.config['eval_every'] == 0:
                self.step(self.val_data_loader, train=False)

                save_dir = os.path.join(self.config['model_dir'], 'weights', f"epoch_{self.epoch}.pt")
                torch.save(self.model.state_dict(), save_dir)
                print(f"Model saved at {save_dir}")

            self.epoch += 1
            self.scheduler.step()


    def step(self, data_loader, train=True, log_writer=True):
        self.model.train() if train else self.model.eval()
        epoch_loss = 0

        total_iou = 0
        total_ade = 0

        for batch in tqdm(data_loader, desc=f"Epoch {self.epoch}/{self.config['epochs']}"):
            for key in batch:
                batch[key] = batch[key].to(self.device)

            if train:
                conditions = augment_data(batch['condition'].float(),
                                          random_length=self.config.get('arbitrary_length_train', False))
            else:
                conds_length = 8
                conditions = augment_data(batch['condition'].float())
                conditions = conditions[:, -conds_length:, :]
            delta_bbox = batch['delta_bbox'].float()

            with torch.amp.autocast('cuda'):
                predicted_delta_bbox = self.model(conditions)
                loss = self.criterion(predicted_delta_bbox, delta_bbox)

            if train:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # MeanIoU and MeanADE
            prev_bbox = conditions[:, -1, :4]
            pred_bbox = prev_bbox + predicted_delta_bbox
            target_bbox = batch['cur_bbox']
            w, h = batch['width'], batch['height']

            orig_pred = original_shape(pred_bbox, w, h)
            orig_target = original_shape(target_bbox, w, h)

            # Calculate IoU
            total_iou += calculate_iou(orig_pred, orig_target)
            total_ade += calculate_ade(orig_pred, orig_target)

            epoch_loss += loss.item()

        print(f"Epoch [{self.epoch}/{self.config['epochs']}],",
              f"Loss: {epoch_loss / len(data_loader):.8f},",
              f"MeanIoU: {total_iou / len(data_loader):.8f},",
              f"MeanADE: {total_ade / len(data_loader):.8f}")

        if log_writer:
            if train:
                self.writer.add_scalar("Loss/train", epoch_loss / len(data_loader), self.epoch)
                self.writer.add_scalar("MeanIoU/train", total_iou / len(data_loader), self.epoch)
                self.writer.add_scalar("MeanADE/train", total_ade / len(data_loader), self.epoch)
                self.writer.add_scalar("Learning rate", self.optimizer.param_groups[0]['lr'], self.epoch)
            else:
                # Show current learning_rate
                for param_group in self.optimizer.param_groups:
                    print(f"Current learning rate: {param_group['lr']:.8f}")
                self.writer.add_scalar("Loss/val", epoch_loss / len(data_loader), self.epoch)
                self.writer.add_scalar("MeanIoU/val", total_iou / len(data_loader), self.epoch)
                self.writer.add_scalar("MeanIoU_FromDelta/val", total_iou / len(data_loader), self.epoch)
                self.writer.add_scalar("MeanADE/val", total_ade / len(data_loader), self.epoch)

    def eval(self):
        """ Evaluate the model """
        import numpy as np
        import cv2
        from tcmp.tracker.bytetrack import BYTETracker
        from tcmp.tracking_utils.log import logger
        from tcmp.tracking_utils.timer import Timer

        def write_results(filename, results, data_type='mot'):
            if data_type == 'mot':
                save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
            elif data_type == 'kitti':
                save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
            else:
                raise ValueError(data_type)

            with open(filename, 'w') as f:
                for frame_id, tlwhs, track_ids in results:
                    if data_type == 'kitti':
                        frame_id -= 1
                    for tlwh, track_id in zip(tlwhs, track_ids):
                        if track_id < 0:
                            continue
                        x1, y1, w, h = tlwh
                        x2, y2 = x1 + w, y1 + h
                        line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                        f.write(line)
            logger.info('save results to {}'.format(filename))

        det_root = self.config['det_dir']
        det_folder_name = '/detections_yolox_x_mix/'
        if det_folder_name in det_root:
            print('Use mix detection')
            img_root = det_root.replace(det_folder_name, '/')
        else:
            img_root = det_root.replace('/detections/', '/')

        seqs = [s for s in os.listdir(det_root)]
        seqs.sort()

        for seq in seqs:
            print(seq)
            # check if results exist
            result_root = self.get_eval_dir
            if os.path.exists(os.path.join(result_root, '{}.txt'.format(seq))):
                print(f"Results for {seq} already exist, skipping...")
                continue
            det_path = os.path.join(det_root, seq)
            img_path = os.path.join(img_root, seq, 'img1')

            info_path = os.path.join(self.config['info_dir'], seq, 'seqinfo.ini')
            seq_info = open(info_path).read()
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

            tracker = BYTETracker(self.config, use_reid=True)
            timer = Timer()
            results = []
            frame_id = 0

            frames = [s for s in os.listdir(det_path)]
            frames.sort()
            imgs = [s for s in os.listdir(img_path) if not s.startswith('.')]
            imgs.sort()

            for i, f in enumerate(frames):
                if frame_id % 100 == 0:
                    logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

                f_path = os.path.join(det_path, f)
                dets = np.loadtxt(f_path, dtype=np.float32, delimiter=',').reshape(-1, 6)[:, 1:6]

                im_path = os.path.join(img_path, imgs[i])
                img = cv2.imread(im_path)
                tag = f"{seq}:{frame_id+1}"

                # track
                online_targets = tracker.update(dets, self.model, frame_id, seq_width, seq_height, tag, img)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

                # save results
                results.append((frame_id + 1, online_tlwhs, online_ids))

                # # visualization
                # online_im = plot_tracking(
                #     img, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                # )
                # cv2.imshow('dancetrack', online_im)
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break

                # vid_writer.write(online_im)

                frame_id += 1

            tracker.dump_cache()
            result_root = self.get_eval_dir
            os.makedirs(result_root, exist_ok=True)
            result_filename = os.path.join(result_root, '{}.txt'.format(seq))
            write_results(result_filename, results)

        if not self.val_set == 'test':
            # Run trackEval
            cmd = f"""python /home/tanndds/my/uav-track/TrackEval/scripts/run_visdrone.py \
            --BENCHMARK {self.config['dataset']} \
            --DO_PREPROC False  \
            --SPLIT_TO_EVAL {self.val_set} \
            --USE_PARALLEL True \
            --TRACKERS_FOLDER "{self.config['model_dir']}/results/" \
            --TRACKERS_TO_EVAL epoch_{self.config["epochs"]}{"_" if self.config.get("postfix", "") != "" else ""}{self.config.get("postfix", "")}
            """
            os.system(cmd)
        else:
            print('Can\'t run trackEval on test set, finish evaluation...')

    def _init_data_loader(self):
        train_path = os.path.join(self.config['data_dir'], 'train')
        val_path = os.path.join(self.config['data_dir'], 'val')
        train_dataset = TrackingDataset(train_path, config=self.config)
        val_dataset = TrackingDataset(val_path, config=self.config)
        print(f"Number of samples in the Train dataset: {len(train_dataset)}")
        print(f"Number of samples in the validation dataset: {len(val_dataset)}")

        val_data_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=4,
            pin_memory=True,
        )
        self.train_data = train_dataset
        self.val_data = val_dataset
        self.val_data_loader = val_data_loader


    def _init_model(self):
        model = DilatedCausalConvNet(config=self.config,
                                     in_channels=8,
                                     residual_channels=64,
                                     skip_channels=64,
                                     out_channels=4,
                                     kernel_size=self.config.get('kernel_size', 2),
                                     num_blocks=self.config.get('num_blocks', 2),
                                     num_layers=self.config.get('num_layers', 4))

        if self.config['resume']:
            if not os.path.exists(self.config['resume']):
                print('Checkpoint file not found, training from scratch...')
            else:
                # get the number of epochs from the checkpoint file
                self.epoch = int(self.config['resume'].split('/')[-1].split('.')[0].split('_')[-1]) + 1
                model.load_state_dict(torch.load(self.config['resume']))
                print('Model loaded from ', self.config['resume'])

        if self.config['eval']:
            weight_dir = os.path.join(self.config['model_dir'], 'weights', f"epoch_{self.config['epochs']}.pt")
            if not os.path.exists(weight_dir):
                print(f'Checkpoint file {weight_dir} not found, evaluation failed...')
                exit()
            else:
                model.load_state_dict(torch.load(weight_dir, weights_only=True))
                print('Model loaded from ', weight_dir)
                self._init_eval_dir()

        self.model = model.to(self.device)
        print('Number of Model\'s parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        with open(os.path.join(self.config['model_dir'], 'model.txt'), 'w') as f:
            f.write(str(model))

    def _init_optimizer(self):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config['epochs'], eta_min=1e-6)
        self.scaler = torch.amp.GradScaler(self.device)

    def _init_model_dir(self):
        if not self.config['model_dir'].startswith('experiments'):
            if not self.config['eval']:
                import re
                if re.match(r'\d{6,8}-\d{6}', self.config['model_dir']):
                    print(f'Found timestamp in {self.config["model_dir"]}, replacing it...')
                    self.config['model_dir'] = re.sub(r'\d{6,8}-\d{6}-', '', self.config['model_dir'], count=1)
                self.config['model_dir'] = self.config['timestamp'] + '-' + self.config['model_dir']
            self.config['model_dir'] = os.path.join('experiments', self.config['model_dir'])


        if not os.path.exists(self.config['model_dir']):
            if self.config['eval']:
                raise FileNotFoundError(f"Model directory {self.config['model_dir']} not found, evaluation failed...")

            print('Create model directory:', self.config['model_dir'])
            os.makedirs(self.config['model_dir'], exist_ok=True)

            with open(os.path.join(self.config['model_dir'], 'config.yml'), 'w') as f:
                yaml.dump(self.config, f)


    def _init_eval_dir(self):
        if 'test' in self.config['config']:
            self.val_set = 'test'
        else:
            self.val_set = 'val'
            if self.config.get('small', False):
                self.val_set = 'val-small'
        eval_dir = os.path.join(self.config['model_dir'],
                                'results',
                                f'{self.config["dataset"]}-{self.val_set}',
                                f'epoch_{self.config["epochs"]}{"_" if self.config.get("postfix", "") != "" else ""}{self.config.get("postfix", "")}',
                                'data')
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir


    def _init_writer(self):
        log_dir = os.path.join('experiments', self.get_exp_name, 'logs')
        self.writer = SummaryWriter(log_dir=log_dir)
        print('Tensorboard logs will be saved at:', log_dir)

    @property
    def get_eval_dir(self):
        return self._init_eval_dir()

    @property
    def get_exp_name(self):
        return self.config['model_dir'].split('experiments/')[1]

