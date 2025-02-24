import torch


def calculate_iou(boxes1, boxes2):
    """
    Calculate the Intersection over Union (IoU) of two sets of bounding boxes.

    Parameters:
    boxes1 (numpy array or tensor): Bounding boxes 1 in the format [batch_size, 4]
    boxes2 (numpy array or tensor): Bounding boxes 2 in the format [batch_size, 4]

    Returns:
    float: Mean IoU value
    """
    box1_x1 = boxes1[:, 0]
    box1_y1 = boxes1[:, 1]
    box1_x2 = boxes1[:, 0] + boxes1[:, 2]
    box1_y2 = boxes1[:, 1] + boxes1[:, 3]

    box2_x1 = boxes2[:, 0]
    box2_y1 = boxes2[:, 1]
    box2_x2 = boxes2[:, 0] + boxes2[:, 2]
    box2_y2 = boxes2[:, 1] + boxes2[:, 3]

    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    inter_area = torch.max(inter_x2 - inter_x1, torch.tensor(0.0)) * torch.max(inter_y2 - inter_y1, torch.tensor(0.0))

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    mean_iou = torch.mean(iou)
    return mean_iou.item()


def calculate_ade(preds, gts):
    """
    Calculate the Average Displacement Error (ADE) between predicted and ground truth bounding boxes.

    Parameters:
    preds (numpy array or tensor): Predicted bounding boxes in the format [batch_size, 4]
    gts (numpy array or tensor): Ground truth bounding boxes in the format [batch_size, 4]

    Returns:
    float: Mean ADE value
    """
    pred_centers = torch.stack([(preds[:, 0] + preds[:, 2] / 2), (preds[:, 1] + preds[:, 3] / 2)], dim=1)
    gt_centers = torch.stack([(gts[:, 0] + gts[:, 2] / 2), (gts[:, 1] + gts[:, 3] / 2)], dim=1)
    displacement = torch.norm(pred_centers - gt_centers, dim=1)
    mean_ade = torch.mean(displacement)
    return mean_ade.item()


def original_shape(boxes, width, height):
    """
    Convert bounding boxes to their original shape based on the width and height.

    Parameters:
    boxes (tensor): Bounding boxes in the format [batch_size, 4]
    width (tensor): Widths for each bounding box in the batch [batch_size]
    height (tensor): Heights for each bounding box in the batch [batch_size]

    Returns:
    tensor: Bounding boxes in their original shape [batch_size, 4]
    """
    boxes_copy = boxes.clone()
    if len(boxes_copy.shape) == 1:
        boxes_copy = boxes_copy.unsqueeze(0)

    boxes_copy[:, 0::2] = boxes_copy[:, 0::2] * width.unsqueeze(1)
    boxes_copy[:, 1::2] = boxes_copy[:, 1::2] * height.unsqueeze(1)
    boxes_copy[:, 0] = boxes_copy[:, 0] - boxes_copy[:, 2] / 2
    boxes_copy[:, 1] = boxes_copy[:, 1] - boxes_copy[:, 3] / 2

    return torch.round(boxes_copy * 10) / 10
