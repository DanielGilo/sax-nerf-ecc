import torch
import numpy as np
from scipy import interpolate
import open3d as o3d
from src.utils.draw_util import plot_rays, plot_cube, plot_camera_pose


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weigh_points(geo, p):
    """
    p in pixel coordinates.
    calculation done in physical units.
    """
    origin = geo.sDetector / 2
    P = p * geo.dDetector - origin
    OP_sq = np.sum(P ** 2, axis=1)
    SP = (geo.DSD ** 2 + OP_sq) ** 0.5
    w = geo.DSD / SP
    return w


def draw_epipolar_setting(geo, rays1, extrinsics1, rays2, extrinsics2):
    pose1 = np.linalg.inv(extrinsics1)
    pose2 = np.linalg.inv(extrinsics2)
    rays_o1 = rays1[:, :3].unsqueeze(dim=1)
    rays_d1 = rays1[:, 3:6].unsqueeze(dim=1)
    rays_o2 = rays2[:, :3].unsqueeze(dim=1)
    rays_d2 = rays2[:, 3:6].unsqueeze(dim=1)
    cube1 = plot_cube(np.zeros((3, 1)), geo.sVoxel[..., np.newaxis])
    cube2 = plot_cube(np.zeros((3, 1)), np.ones((3, 1)) * geo.DSO * 2)
    prays1 = plot_rays(rays_d1.cpu().detach().numpy(), rays_o1.cpu().detach().numpy(), 2)
    prays2 = plot_rays(rays_d2.cpu().detach().numpy(), rays_o2.cpu().detach().numpy(), 2)
    poseray1 = plot_camera_pose(pose1)
    poseray2 = plot_camera_pose(pose2)
    o3d.visualization.draw_geometries([cube1, cube2, prays1, prays2, poseray1, poseray2])


def world_rays_from_image_lines(x, y, geo, near, far, extrinsics):
    pose = torch.Tensor(np.linalg.inv(extrinsics)).to(device)
    W, H = geo.nDetector
    x_c = torch.Tensor((x - W / 2) * geo.dDetector[0])
    y_c = torch.Tensor((y - H / 2) * geo.dDetector[1])
    dirs_c = torch.stack([x_c / geo.DSD, y_c / geo.DSD, torch.ones_like(x_c)], -1).to(device)
    rays_d_w = torch.sum(torch.matmul(pose[:3, :3], dirs_c[..., None]).to(device), -1)  # pose[:3, :3] *
    rays_o_w = pose[:3, -1].expand(rays_d_w.shape)
    rays = torch.concat([rays_o_w.to(device), rays_d_w], dim=-1)
    return torch.cat([rays, torch.ones_like(rays[..., :1]) * near, torch.ones_like(rays[..., :1]) * far], dim=-1)


def get_matching_epipolar_lines(intrinsics, extrinsics1, extrinsics2, P_w):
    F = torch.Tensor(get_fundamental_matrix(intrinsics, extrinsics1.numpy(), extrinsics2.numpy()))
    p1_h = intrinsics @ extrinsics1.numpy() @ P_w
    p2_h = intrinsics @ extrinsics2.numpy() @ P_w
    l1 = F.T @ p2_h
    l2 = F @ p1_h
    return l1, l2


def get_line_in_image(slope, intercept, h, W, n_samples):
    x_initial = np.linspace(0.5, W - 0.5, n_samples)
    y_initial = x_initial * slope + intercept
    in_img_indices = (0.5 < y_initial - h) * (y_initial + h < (W - 0.5))
    x_in_img = np.linspace(x_initial[in_img_indices][0], x_initial[in_img_indices][-1], n_samples)
    y_in_img = x_in_img * slope + intercept
    return x_in_img, y_in_img


def get_epipolar_constraint_rays(intrinsics, extrinsics1, extrinsics2, geo, P_w, near, far, h, n_samples):
    W, H = geo.nDetector
    l1, l2 = get_matching_epipolar_lines(intrinsics, extrinsics1, extrinsics2, P_w)
    slope_1, intercept_1 = (-l1[0] / l1[1]).item(), (-l1[2] / l1[1]).item()
    slope_2, intercept_2 = (-l2[0] / l2[1]).item(), (-l2[2] / l2[1]).item()

    x1, y1 = get_line_in_image(slope_1, intercept_1, h, W, n_samples)
    x2, y2 = get_line_in_image(slope_2, intercept_2, h, W, n_samples)

    epiline_1_neigh_up_rays = world_rays_from_image_lines(x1, y1 + h, geo, near, far, extrinsics1)
    epiline_1_neigh_down_rays = world_rays_from_image_lines(x1, y1 - h, geo, near, far,
                                                            extrinsics1)
    epiline_2_neigh_up_rays = world_rays_from_image_lines(x2, y2 + h, geo, near, far, extrinsics2)
    epiline_2_neigh_down_rays = world_rays_from_image_lines(x2, y2 - h, geo, near, far,
                                                            extrinsics2)

    # draw_epipolar_setting(geo, epiline_1_neigh_up_rays, extrinsics1, epiline_2_neigh_up_rays, extrinsics2)

    dx1 = x1[1] - x1[0]
    dx2 = x2[1] - x2[0]
    y_dx1 = dx1 * slope_1
    y_dx2 = dx2 * slope_2
    dr1 = (dx1 ** 2 + y_dx1 ** 2) ** 0.5
    dr2 = (dx2 ** 2 + y_dx2 ** 2) ** 0.5

    w1_up = weigh_points(geo, list(zip(x1, y1 + h))) * dr1
    w1_down = weigh_points(geo, list(zip(x1, y1 - h))) * dr1
    w2_up = weigh_points(geo, list(zip(x2, y2 + h))) * dr2
    w2_down = weigh_points(geo, list(zip(x2, y2 - h))) * dr2

    # w1_up = weigh_points_by_grid_sample(cosine_weights, p=list(zip(x1, y1 + h)), geo=geo) * dr1
    # w1_down = weigh_points_by_grid_sample(cosine_weights, p=list(zip(x1, y1 - h)), geo=geo) * dr1
    # w2_up = weigh_points_by_grid_sample(cosine_weights, p=list(zip(x2, y2 + h)), geo=geo) * dr2
    # w2_down = weigh_points_by_grid_sample(cosine_weights, p=list(zip(x2, y2 - h)), geo=geo) * dr2

    rays = torch.stack([epiline_1_neigh_up_rays, epiline_1_neigh_down_rays, epiline_2_neigh_up_rays, epiline_2_neigh_down_rays])
    weights = torch.tensor(np.array([w1_up, w1_down, w2_up, w2_down]))

    return rays, weights




#### functions adapted from: https://github.com/tatakai1/EVENeRF/blob/main/evenerf/data_loaders/data_verifier.py
def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def get_fundamental_matrix(intrinsics, extrinsics1, extrinsics2):
    relative_pose = extrinsics2.dot(np.linalg.inv(extrinsics1))
    R = relative_pose[:3, :3]
    T = relative_pose[:3, 3]
    tx = skew(T)
    E = np.dot(tx, R)
    F = np.linalg.inv(intrinsics[:3, :3]).T.dot(E).dot(np.linalg.inv(intrinsics[:3, :3]))
    return F



def generate_random_3D_point(geo):
    """
    Generates random 3D point in homogenous world coordinates, such that when projected to the detector it will
     be contained in it.
    """
    z_max = ((geo.sDetector[1] - (4 * geo.dDetector[1])) / 2) * (geo.DSO / geo.DSD)
    z_min = -z_max
    z = np.random.uniform(low=z_min, high=z_max)
    return torch.Tensor([0.0, 0.0, z, 1])


def angle2extrinsics(angle, dset):
    return torch.Tensor(np.linalg.inv(dset.angle2pose(dset.geo.DSO, angle)))


def get_random_angle_pair():
    angle_0 = np.random.uniform(low=0, high=2 * np.pi)
    d_theta = np.random.uniform(low=0.05, high=np.pi - 0.05)  # avoid opposite views
    if (angle_0 + d_theta) < (2 * np.pi):
        angle_1 = angle_0 + d_theta
    else:
        angle_1 = angle_0 + d_theta - (2 * np.pi)
    return angle_0, angle_1


def get_random_epipolar_rays(dset, n_epipolar, h, n_samples):
    angle_0, angle_1 = get_random_angle_pair()
    extrinsics_1 = angle2extrinsics(angle_0, dset)
    extrinsics_2 = angle2extrinsics(angle_1, dset)

    P_w = generate_random_3D_point(dset.geo)
    rays, weights = get_epipolar_constraint_rays(
        dset.intrinsics_mat, extrinsics_1, extrinsics_2, dset.geo, P_w, dset.near,
        dset.far, h=h, n_samples=n_samples)
    return rays, weights






### Functions previously used. TODO: delete!

"""
def sample_image_along_line(img, slope, intercept, n_samples):
    
    slope, intercept and dx in pixel coordinates
    
    img = img.cpu().numpy()
    (H, W) = img.shape
    x_orig = np.arange(H) + 0.5
    y_orig = np.arange(W) + 0.5
    interp_img = interpolate.RegularGridInterpolator((x_orig, y_orig), img, method='slinear')
    x_new = np.linspace(0.5, W - 0.5, n_samples)
    y_new = x_new * slope + intercept
    in_img_indices = (0.5 < y_new) * (y_new < (W - 0.5))
    line = interp_img((y_new[in_img_indices], x_new[in_img_indices]))
    dx = x_new[1] - x_new[0]
    y_dx = dx * slope
    dr = (dx ** 2 + y_dx ** 2) ** 0.5
    return line, dr


def grad_of_integral_over_line(img, slope, intercept, n_samples, h):
    
    slope, intercept, dx and h in pixel coordinates
    
    upper_neigh, dr1 = sample_image_along_line(img, slope, intercept + h, n_samples)
    lower_neigh, dr2 = sample_image_along_line(img, slope, intercept - h, n_samples)
    return (np.sum(upper_neigh * dr1) - np.sum(lower_neigh * dr2)) / (2 * h)


def get_extrinsics_pair_from_dset(dset):
    select_indices = np.random.choice(len(dset), size=2, replace=False)
    return dset.extrinsics_mats[select_indices[0]], dset.extrinsics_mats[select_indices[1]], select_indices



def eval_epipolar_consistency(intrinsics, extrinsics1, extrinsics2, geo, proj1, proj2, n_samples, h, n_eval=20,
                              P_ws=None):
    grads_se = []
    rand_grads_se = []
    z = []
    if P_ws is None:
        P_ws = [generate_random_3D_point(geo) for i in range(n_eval)]
    for P_w in P_ws:
        l1, l2 = get_matching_epipolar_lines(intrinsics, extrinsics1, extrinsics2, P_w)
        slope_1, intercept_1 = (-l1[0] / l1[1]).item(), (-l1[2] / l1[1]).item()
        slope_2, intercept_2 = (-l2[0] / l2[1]).item(), (-l2[2] / l2[1]).item()
        grad_1 = grad_of_integral_over_line(weigh_proj(proj1, geo), slope_1, intercept_1, n_samples, h)
        grad_2 = grad_of_integral_over_line(weigh_proj(proj2, geo), slope_2, intercept_2, n_samples, h)
        grad_rand_intercept = grad_of_integral_over_line(weigh_proj(proj2, geo), slope_2,
                                                         np.random.uniform(low=1, high=geo.nDetector[1] - 1), n_samples,
                                                         h)
        grads_se.append((grad_1 - grad_2) ** 2)
        rand_grads_se.append((grad_rand_intercept - grad_2) ** 2)
        z.append(P_w[2])
    return np.mean(grads_se), np.max(grads_se), np.max(rand_grads_se), P_ws


def weigh_points_by_grid_sample(weights, p, geo):
    p = np.array(p)
    input = weights.unsqueeze(0).unsqueeze(0) # [H,W] to [1, 1, H, W] for 2D weight grid
    normalized_p = (p / (geo.nDetector/ 2)) - 1 # normalize query for [-1, 1] range
    query = torch.empty(1, p.shape[0], 2).to(device) # not sure about p.shape, it should be [H_out, W_out, 2]
    query[:, :, 0] = torch.tensor(normalized_p[:, 0]) # x
    query[:, :, 1] = torch.tensor(normalized_p[:, 1]) # y
    query = query.unsqueeze(0) # [1, H_out, W_out, 2]
    weights = torch.nn.functional.grid_sample(input=input,grid=query, mode='bilinear') # [1, 1, H_out, W_out]
    return weights[0,0,0,:].cpu()
"""