import torch

def jaccard_index(pred_labels, target_labels, device="cpu"):
    """Computes the jaccard index given prediction and target labels.

    Args:
        pred_labels (torch.Tensor): The segmentation predictions of size (B, H, W) where B is batch size, H is height and W is width.
            The dimensions have to be the same as the target_labels
            Assumes that all entries are either 0 or 1.
        target_labels (torch.Tensor): The target segmentation of size (B, H, W) where B is batch size, H is height and W is width.
            The dimensions have to be the sames as pred_labels
        device (str, optional): The device to which to cast the result to. Is only needed if both sets are empty. Defaults to "cpu".

    Returns:
        torch.Tensor: The mean jaccard index over the batch size
    """
    # Cast to same type
    pred_labels = pred_labels.int()
    target_labels = target_labels.int()

    # Compute union and intersection
    union = torch.bitwise_or(pred_labels, target_labels).sum()
    intersection = torch.bitwise_and(pred_labels, target_labels).sum()

    # Except when union is 0 the Jaccard index is one
    if union == 0:
        return torch.tensor(1.0, device=device)
    else:
        return intersection / union

