import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class TrackingDataset(Dataset):
    def __init__(self, path, config=None):
        self.config = config.copy()
        self.path = path
        self.set = path.split("/")[-1]

        tic = time.time()
        self.aug_random_var = 0.001
        self.config["augment_data"] = self.config.get("augment_data", False)
        # Force disable data augmentation for val set
        if self.set == 'val':
            print("Disabling data augmentation for validation set.")
            self.config["augment_data"] = False

        # Default interval is 5
        self.interval = self.config.get("interval", 5)

        self.trackers = {}
        self.data = []  

        if not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a valid directory.")

        self.seqs = [s for s in os.listdir(path) if not s.startswith('.') and "gt_t" not in s]
        for seq in self.seqs:
            trackerPath = os.path.join(path, seq, "img1/*.txt")
            self.trackers[seq] = sorted(glob.glob(trackerPath))

            for pa in self.trackers[seq]:
                gt = np.loadtxt(pa, dtype=np.float32)
                self.precompute_data(seq, gt)  # Precompute data for this sequence

        print(f"Loaded {len(self.data)} items in {time.time() - tic:.2f}s")

    def precompute_data(self, seq, track_gt):
        """
        Precompute and store data for the dataset.
        
        Parameters
        ----------
            seq: str
                The sequence name.
            track_gt: ndarray
                Ground truth data for the sequence.
        """
        if len(track_gt.shape) < 2:
            return
        boxes = track_gt[:, 2:6]
        deltas = np.diff(boxes, axis=0)
        conds = np.concatenate([boxes[:-1], deltas], axis=1)

        for init_index in range(0, len(track_gt) - self.interval - 1):
            curr_idx = init_index + self.interval

            data_item = {
                "cur_gt": track_gt[curr_idx],  # ndarray (9, )
                "cur_bbox": track_gt[curr_idx, 2:6],  # ndarray (4, )
                "condition": conds[init_index:curr_idx],  # ndarray (interval, 8)
                "delta_bbox": deltas[curr_idx, :],  # ndarray (4, )
                "width": track_gt[curr_idx, 7],  # float
                "height": track_gt[curr_idx, 8],  # float
            }
            self.data.append(data_item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def show_image(self, index):
        """Display the image at the given index using PIL."""
        # Get the image path from the dataset
        image_path = self.data[index]['image_path']
        
        # Open the image using PIL
        img = Image.open(image_path)

        # Display the image with matplotlib
        plt.imshow(img)
        plt.axis("off")  # Hide axis for a cleaner look
        plt.show()
        
        # Display the image
        img.show()

def randomly_truncate_boxes(boxes):
    trajectory_length = boxes.shape[1]
    random_number = torch.randint(0, trajectory_length-4, (1,)).item()
    return boxes[:, random_number:, :]

def augment_data(boxes, aug_random_var=0.001, random_length=False):
    noise = torch.randn_like(boxes[:, :, :4]) * aug_random_var
    boxes[:, :, :4] += noise.to(boxes.device)
    boxes[:, 1:, 4:] = boxes[:, 1:, :4] - boxes[:, :-1, :4]

    if random_length:
        return randomly_truncate_boxes(boxes)
    return boxes


def custom_collate_fn(batch):
    for sample in batch:
        if 'image_path' in sample:
            del sample['image_path']
    return torch.utils.data.default_collate(batch)



# Load the configuration file
import yaml


if __name__ == '__main__':
    config_path = '../configs/default.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config['augment_data'] = False

    dataset = TrackingDataset("/home/tanndds/my/datasets/dancetrack/trackers_gt_t/train", config)
    # print(len(dataset))
    # n = 100_000_000
    # tic = time.time()
    # for _ in range(n):
    #     d = dataset[0]
    #
    # print(f"Time taken to load {n} samples: {time.time() - tic:.2f}s")

    # create dataloader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True, collate_fn=custom_collate_fn)

    # iterate over the dataset
    runtime = 0
    for i, data in enumerate(train_loader):
        print(data['condition'].shape)
        # tic = time.time()
        data['condition'] = augment_data(data['condition'])
        # runtime += time.time() - tic

    print(f"Time taken to augment the data: {runtime:.4f}s")
