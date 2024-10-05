import torch


def calc_epipolar_loss(loss, projs, weights, h, w, n_epipolar):
    """
        Calculate epipolar consistency loss.
    """
    # projs = projs.reshape(n_epipolar, 4, -1)
    # l1_upper = projs[:, 0, :]
    # l1_lower = projs[:, 1, :]
    # l2_upper = projs[:, 2, :]
    # l2_lower = projs[:, 3, :]
    # l1_integral_grad = (torch.sum(l1_upper * drs[:,0], dim=1) - torch.sum(l1_lower * drs[:,1], dim=1)) / (2 * h)
    # l2_integral_grad = (torch.sum(l2_upper * drs[:,0], dim=1) - torch.sum(l2_lower * drs[:,1], dim=1)) / (2 * h)

    l1_upper, l1_lower, l2_upper, l2_lower = projs
    w1_upper, w1_lower, w2_upper, w2_lower = weights

    l1_integral_grad = (torch.sum(l1_upper * w1_upper) - torch.sum(l1_lower * w1_lower)) / (2 * h)
    l2_integral_grad = (torch.sum(l2_upper * w2_upper) - torch.sum(l2_lower * w2_lower)) / (2 * h)

    loss_epipolar = torch.mean((l1_integral_grad - l2_integral_grad) ** 2)

    loss["loss"] += w * loss_epipolar
    loss["loss_epipolar"] = loss_epipolar
    return loss


def calc_mse_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss_mse = torch.mean((x-y)**2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
    return loss

def calc_mse_loss_raw(loss, x, y, k = 1):
    """
    Calculate mse loss for raw.
    """
    # Compute loss for raw
    loss_mse_raw = torch.mean((x-y)**2)
    loss["loss"] += k * loss_mse_raw
    loss["loss_mse_raw"] = loss_mse_raw
    return loss

def calc_tv_loss(loss, x, k):
    """
    Calculate total variation loss.
    Args:
        x (n1, n2, n3, 1): 3d density field.
        k: relative weight
    """
    n1, n2, n3 = x.shape
    tv_1 = torch.abs(x[1:,1:,1:]-x[:-1,1:,1:]).sum()
    tv_2 = torch.abs(x[1:,1:,1:]-x[1:,:-1,1:]).sum()
    tv_3 = torch.abs(x[1:,1:,1:]-x[1:,1:,:-1]).sum()
    tv = (tv_1+tv_2+tv_3) / (n1*n2*n3)
    loss["loss"] += tv * k
    loss["loss_tv"] = tv * k
    return loss



