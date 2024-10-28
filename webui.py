import time
import numpy as np
import torch
import torchvision
from scene import Scene
from scene.cameras import Camera
from scene.colmap_loader import rotmat2qvec
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

import viser
import viser.transforms as tf
from dataclasses import dataclass, field
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from gaussian_renderer import render

import numpy as np
import torch
import random

import math


def get_device():
    return torch.device(f"cuda")


class Simple_Camera:
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        h,
        w,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        qvec=None,
    ):
        super(Simple_Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.qvec = qvec

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.image_width = w
        self.image_height = h

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def HW_scale(self, h, w):
        return Simple_Camera(
            self.colmap_id, self.R, self.T, self.FoVx, self.FoVy, h, w, self.image_name, self.uid, qvec=self.qvec
        )


class WebUI:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.render_cameras = None

        self.port = 8084
        self.server = viser.ViserServer(port=self.port)
        self.add_theme()
        self.draw_flag = True
        with self.server.add_gui_folder("Render Setting"):
            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=4096, step=2, initial_value=2048
            )

            self.FoV_slider = self.server.add_gui_slider("FoV Scaler", min=0.2, max=2, step=0.1, initial_value=1)
            self.scale_slider = self.server.add_gui_slider("Gaussian Scale", min=0.1, max=1, step=0.1, initial_value=1)

            self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)
            self.renderer_output = self.server.add_gui_dropdown("Renderer Output", ["comp_rgb"])
            self.save_button = self.server.add_gui_button("Save Gaussian")

            self.frame_show = self.server.add_gui_checkbox("Show Frame", initial_value=True)
            self.train = self.server.add_gui_checkbox("Train", initial_value=True)

        with torch.no_grad():
            self.frames = []
            random.seed(0)
            cams = self.scene.getTrainCameras()
            frame_index = random.sample(range(0, len(cams)), min(len(cams), 20))
            for i in frame_index:
                self.make_one_camera_pose_frame(cams[i])

        @self.frame_show.on_update
        def _(_):
            for frame in self.frames:
                frame.visible = self.frame_show.value
            self.server.world_axes.visible = self.frame_show.value

    def make_one_camera_pose_frame(self, cam: Camera):
        # wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
        # position = -cam.R.T @ cam.T

        T_world_camera = tf.SE3.from_rotation_and_translation(tf.SO3.from_matrix(cam.R.T), cam.T).inverse()
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        # breakpoint()
        frame = self.server.add_frame(
            f"/colmap/frame_{cam.colmap_id}",
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=False,
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(tf.SO3(client.camera.wxyz), client.camera.position)

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(T_current_target.log() * j / 4.0)

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):

            def begin_trans(client):
                assert client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )

                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(T_current_target.log() * j / 4.0)

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            self.begin_call = begin_trans

    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        if self.render_cameras is None:
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
            self.render_cameras = self.scene.getTrainCameras()
            self.begin_call(list(self.server.get_clients().values())[0])

        viser_cam = list(self.server.get_clients().values())[0].camera
        # viser_cam.up_direction = tf.SO3(viser_cam.wxyz) @ np.array([0.0, -1.0, 0.0])
        # viser_cam.look_at = viser_cam.position
        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position
        # print(viser_cam.position)
        # T = viser_cam.position
        if self.render_cameras is None:
            fovy = viser_cam.fov * self.FoV_slider.value
        else:
            fovy = self.render_cameras[0].FoVy * self.FoV_slider.value

        fovx = 2 * math.atan(math.tan(fovy / 2) * self.aspect)
        # fovy = self.render_cameras[0].FoVy
        # fovx = self.render_cameras[0].FoVx
        # math.tan(self.render_cameras[0].FoVx / 2) / math.tan(self.render_cameras[0].FoVy / 2)
        # math.tan(fovx/2) / math.tan(fovy/2)

        # print(viser_cam.wxyz)

        # aspect = viser_cam.aspect
        width = int(self.resolution_slider.value)
        height = int(width / self.aspect)
        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)

    @torch.no_grad()
    def prepare_output_image(self, output):
        out_key = self.renderer_output.value
        out_img = output[out_key]  # C H W
        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)

        self.renderer_output.options = list(output.keys())
        return out_img.cpu().moveaxis(0, -1).numpy().astype(np.uint8)

    def render_loop(self, gaussians, pipline, background):
        while True:
            # if self.viewer_need_update:
            if self.camera is not None:
                render_package = render(self.camera, gaussians, pipline, background, self.scale_slider.value)
                self.update_viewer({"comp_rgb": render_package["render"]})
            torch.cuda.empty_cache()
            time.sleep(1e-2)

    @torch.no_grad()
    def update_viewer(self, outputs):
        out = self.prepare_output_image(outputs)
        self.server.set_background_image(out, format="jpeg")

    def add_theme(self):
        buttons = (
            TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://github.com/buaacyw/GaussianEditor/blob/master/docs/webui.md",
            ),
            TitlebarButton(text="Github", icon="GitHub", href="https://github.com/buaacyw/GaussianEditor"),
            TitlebarButton(text="Yiwen Chen", icon=None, href="https://buaacyw.github.io/"),
            TitlebarButton(
                text="Zilong Chen", icon=None, href="https://scholar.google.com/citations?user=2pbka1gAAAAJ&hl=en"
            ),
        )
        image = TitlebarImage(
            image_url_light="https://github.com/buaacyw/gaussian-editor/raw/master/static/images/logo.png",
            image_alt="GaussianEditor Logo",
            href="https://buaacyw.github.io/gaussian-editor/",
        )
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image)
        brand_color = self.server.add_gui_rgb("Brand color", (7, 0, 8), visible=False)

        self.server.configure_theme(titlebar_content=titlebar_theme, show_logo=True, brand_color=brand_color.value)
