#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import random
import math
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.loss_utils import image2canny

class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        K,
        image,
        orig_w,
        orig_h,
        depth,
        depth_weight,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.canny_mask = image2canny(self.original_image.permute(1,2,0), 50, 150, isEdge1=False).detach().to(self.data_device)
        self.original_depth = depth.to(self.data_device) if depth is not None else None
        self.original_depth_weight = depth_weight.to(self.data_device) if depth_weight is not None else None
        self.original_mask = gt_alpha_mask.to(self.data_device) if gt_alpha_mask is not None else None
        # self.original_mask = gt_alpha_mask>0.9 if gt_alpha_mask is not None else None
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.original_width = orig_w
        self.original_height = orig_h

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            if depth is not None:
                self.original_depth *= gt_alpha_mask[0].to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        def getProjectionMatrixK(znear, zfar, K):
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            tanHalfFovY = math.tan((self.FoVy / 2))
            tanHalfFovX = math.tan((self.FoVx / 2))

            top = cy * znear / fy
            bottom = -2 * tanHalfFovY * znear + cy * znear / fy  # # cx_default = w // 2
            right = 2 * tanHalfFovX * znear - cx * znear / fx
            left = -cx * znear / fx

            P = torch.zeros(4, 4)

            z_sign = 1.0

            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = -z_sign * (right + left) / (right - left)
            P[1, 2] = z_sign * (top + bottom) / (top - bottom)
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        if K is not None:
            self.projection_matrix = getProjectionMatrixK(znear=self.znear, zfar=self.zfar, K=K).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


class RandomCamera:
    def __init__(self, dataset_cameras, world_center, max_point_span, trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        self.FoVxs = [camera.FoVx for camera in dataset_cameras]
        self.FoVys = [camera.FoVy for camera in dataset_cameras]
        self.image_widths = [camera.image_width for camera in dataset_cameras]
        self.image_heights = [camera.image_height for camera in dataset_cameras]

        self.world_center = world_center
        self.max_point_span = max_point_span

        self.trans = trans
        self.scale = scale

        centers = np.array([camera.camera_center.cpu().numpy() for camera in dataset_cameras])
        directions = (centers - world_center) / np.linalg.norm(centers - world_center, axis=1, keepdims=True)
        lengths = np.linalg.norm(centers - world_center, axis=1)
        self.voting_vectors = np.concat([directions, np.diag([1.0] * 3)])
        self.offset_range = (lengths.min(), lengths.max() * 2)

        # direction = directions[0]
        # center = centers[0]
        # cam = dataset_cameras[0]
        # lookat_direction = -direction

        # up_direction = np.array([0.0, 1.0, 0.0])
        # camera_right = np.cross(up_direction, direction)
        # camera_up = np.cross(lookat_direction, camera_right)

        # R_c = np.stack([camera_right, camera_up, lookat_direction], axis=1)
        # Rt = np.zeros((4, 4))
        # Rt[:3, :3] = cam.R.T
        # Rt[:3, 3] = cam.T
        # Rt[3, 3] = 1.0

        # C2W = np.linalg.inv(Rt)

        self.randomize()

    def randomize(self):
        self.image_width = random.choice(self.image_widths)
        self.image_height = random.choice(self.image_heights)
        self.FoVx = random.choice(self.FoVxs)
        self.FoVy = random.choice(self.FoVys)

        self.zfar = 100.0
        self.znear = 0.01

        direction = np.sum(np.random.randn(len(self.voting_vectors)).reshape(-1, 1) * self.voting_vectors, axis=0)
        while np.linalg.norm(direction) == 0:
            direction = np.sum(np.random.randn(len(self.voting_vectors)).reshape(-1, 1) * self.voting_vectors, axis=0)
        direction /= np.linalg.norm(direction)

        distance = random.uniform(*self.offset_range)
        self.camera_center = self.world_center + distance * direction

        lookat_offset = np.random.randn(3)
        while np.linalg.norm(lookat_offset) == 0:
            lookat_offset = np.random.randn(3)

        lookat_offset /= np.linalg.norm(lookat_offset)

        lookat_position = self.world_center + lookat_offset * np.random.uniform(0, self.max_point_span)
        while np.linalg.norm(lookat_position - self.camera_center) == 0:
            lookat_position = self.world_center + lookat_offset * np.random.uniform(0, self.max_point_span)

        lookat_direction = (lookat_position - self.camera_center) / np.linalg.norm(lookat_position - self.camera_center)

        up_direction = np.array([0.0, 1.0, 0.0])
        camera_right = np.cross(lookat_direction, up_direction)
        camera_up = np.cross(lookat_direction, camera_right)

        R_c = np.stack([camera_right, camera_up, lookat_direction], axis=1)
        C = self.camera_center
        C2W = np.zeros((4, 4))
        C2W[:3, :3] = R_c
        C2W[:3, 3] = C
        C2W[3, 3] = 1.0

        def getWorld2ViewFromC2W(C2W, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
            cam_center = C2W[:3, 3]
            cam_center = (cam_center + translate) * scale
            C2W[:3, 3] = cam_center
            Rt = np.linalg.inv(C2W)
            return np.float32(Rt)

        self.world_view_transform = (
            torch.tensor(getWorld2ViewFromC2W(C2W, self.trans, self.scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def randomized_casual(self):
        self.image_width = random.choice(self.image_widths)
        self.image_height = random.choice(self.image_heights)
        self.FoVx = random.choice(self.FoVxs)
        self.FoVy = random.choice(self.FoVys)

        self.zfar = 100.0
        self.znear = 0.01

        voting_vectors = np.diag([1.0] * 3)

        direction = np.sum(np.random.randn(len(voting_vectors)).reshape(-1, 1) * voting_vectors, axis=0)
        while np.linalg.norm(direction) == 0:
            direction = np.sum(np.random.randn(len(voting_vectors)).reshape(-1, 1) * voting_vectors, axis=0)
        direction /= np.linalg.norm(direction)

        distance = random.uniform(*self.offset_range)
        self.camera_center = self.world_center + distance * direction
        lookat_direction = direction * (np.random.randint(0, 2) * 2 - 1)

        up_direction = np.array([0.0, 1.0, 0.0])
        camera_right = np.cross(up_direction, direction)
        camera_up = np.cross(lookat_direction, camera_right)

        R_c = np.stack([camera_right, camera_up, lookat_direction], axis=1)
        C = self.camera_center
        C2W = np.zeros((4, 4))
        C2W[:3, :3] = R_c
        C2W[:3, 3] = C
        C2W[3, 3] = 1.0

        def getWorld2ViewFromC2W(C2W, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
            cam_center = C2W[:3, 3]
            cam_center = (cam_center + translate) * scale
            C2W[:3, 3] = cam_center
            Rt = np.linalg.inv(C2W)
            return np.float32(Rt)

        self.world_view_transform = (
            torch.tensor(getWorld2ViewFromC2W(C2W, self.trans, self.scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
