import numpy as np
import torch
from torch import nn as nn


class BasePositionPredictor(nn.Module):
    def __init__(self, config):
        super(BasePositionPredictor, self).__init__()
        self.config = config

    def forward(self, x):
        raise NotImplementedError

    def augment_data(self, boxes):
        """Augment the data item, by offset the boxes by a small random value."""
        if len(boxes.shape) == 2:
            boxes = boxes.unsqueeze(0)
        xywh = boxes[:, :, :4]
        xywh += torch.rand_like(xywh) * 5e-3

        boxes[:, 1:, :4] = xywh[:, 1:]
        delta_xywh = boxes[:, :, 4:]
        delta_xywh[:, 1:, :] = xywh[:, 1:] - xywh[:, :-1]
        if len(boxes) > 1:
            delta_xywh[:, 0, :] = delta_xywh[:, 1, :]
        boxes[:, :, 4:] = delta_xywh
        return boxes

    def generate(self, conditions, img_w, img_h, **kwargs):
        cond_encodeds = []
        can_work_with_arbitrary_length = kwargs.get("can_work_with_arbitrary_length", False)

        all_same_length = all(len(condition) == len(conditions[0]) for condition in conditions)

        if can_work_with_arbitrary_length:
            if all_same_length:
                for i in range(len(conditions)):
                    tmp_c = torch.tensor(np.array(conditions[i]), dtype=torch.float, device="cuda")
                    # normalize the condition
                    tmp_c[:, 0::2] = tmp_c[:, 0::2] / img_w
                    tmp_c[:, 1::2] = tmp_c[:, 1::2] / img_h
                    tmp_c = tmp_c.unsqueeze(0)

                    cond_encodeds.append(tmp_c)
                cond_encodeds = torch.cat(cond_encodeds).to("cuda")

                with torch.no_grad():
                    track_pred = self.forward(cond_encodeds)
                return track_pred.cpu().detach().numpy()
            else:
                track_pred = []
                for i in range(len(conditions)):
                    tmp_c = torch.tensor(np.array(conditions[i]), dtype=torch.float, device="cuda")
                    # normalize the condition
                    tmp_c[:, 0::2] = tmp_c[:, 0::2] / img_w
                    tmp_c[:, 1::2] = tmp_c[:, 1::2] / img_h
                    tmp_c = tmp_c.unsqueeze(0)

                    with torch.no_grad():
                        track_pred.append(self.forward(tmp_c).cpu())

                track_pred = torch.cat(track_pred, dim=0).numpy()
                return track_pred
        else:
            for i in range(len(conditions)):
                tmp_c = torch.tensor(np.array(conditions[i]), dtype=torch.float, device="cuda")
                # normalize the condition
                tmp_c[:, 0::2] = tmp_c[:, 0::2] / img_w
                tmp_c[:, 1::2] = tmp_c[:, 1::2] / img_h

                # pad the condition to the interval
                if len(tmp_c) < self.config['interval']:
                    pad_conds = tmp_c[-1].repeat((self.config['interval'] - len(tmp_c), 1)).to("cuda")
                    pad_conds = self.augment_data(pad_conds)
                    tmp_c = torch.cat((tmp_c.unsqueeze(0), pad_conds), dim=1)
                else:
                    tmp_c = tmp_c[-self.config['interval']:]
                    tmp_c = tmp_c.unsqueeze(0)
                cond_encodeds.append(tmp_c)
            cond_encodeds = torch.cat(cond_encodeds).to("cuda")

            with torch.no_grad():
                track_pred = self.forward(cond_encodeds)
            return track_pred.cpu().detach().numpy()
