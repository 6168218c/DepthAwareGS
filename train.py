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

import os
import sys
import uuid
from tqdm import tqdm
from random import randint, choices
import cv2
import numpy as np
import torch
import torch.fft
import torchvision

from webui import WebUI
import time

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from scene.cameras import RandomCamera
from gaussian_renderer import render, network_gui
from render import depth_colorize_with_mask

from utils.loss_utils import l1_loss, l2_loss, nearMean_map
from utils.image_utils import psnr, normalize_depth
from utils.loss_utils import ssim
from utils.general_utils import get_expon_lr_func
from utils.fft_utils import get_fft_kernel
from lpipsPyTorch import lpips

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    usedepth=False,
    use_projector=False,
    use_fft_smooth=False,
    use_anchor_loss=False,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_depthloss_for_log, prev_depthloss, deploss = 0.0, 1e2, torch.zeros(1)
    apply_anchor_regularization = False
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    train_cameras = scene.getTrainCameras()
    if use_projector:
        train_cameras = train_cameras
        projector_camera = train_cameras[-1]
    random_camera = RandomCamera(scene.getTrainCameras(), scene.world_center, scene.max_point_distance)
    webgui = WebUI(scene)

    @torch.no_grad()
    def render_webui():
        if len(webgui.server.get_clients()) > 0:
            training = False
            while not training:
                camera = webgui.camera
                if camera is not None:
                    # proj_cam = [cam for cam in scene.getTestCameras() if cam.colmap_id == 2][0]
                    render_package = render(camera, gaussians, pipe, background, webgui.scale_slider.value)
                    # proj_package = render(proj_cam, gaussians, pipe, background, 1.0)
                    random_package = render(random_camera, gaussians, pipe, background, webgui.scale_slider.value)

                    hf_kernel = get_fft_kernel(random_camera)
                    image_weights = torch.fft.fft2(random_package["render"])
                    image_weights = torch.fft.fftshift(image_weights, dim=(-2, -1))
                    image_weights = image_weights * hf_kernel
                    image_weights = torch.fft.ifftshift(image_weights, dim=(-2, -1))
                    image_weights = torch.abs(torch.fft.ifft2(image_weights))

                    ui_images = {
                        "comp_rgb": render_package["render"],
                        "random": random_package["render"],
                        "random_fft": image_weights,
                        # "projector": proj_package["render"],
                        # "projector_gt": proj_cam.original_image,
                        # "depth": torch.tensor(
                        #     depth_colorize_with_mask(
                        #         render_package["depth"].detach().cpu().unsqueeze(0), dmindmax=(20.0, 130.0)
                        #     ).squeeze(),
                        #     dtype=torch.float32,
                        # ).moveaxis(-1, 0),
                    }
                    webgui.update_viewer(ui_images)
                training = webgui.train.value
                if not training:
                    torch.cuda.empty_cache()
                    time.sleep(1e-2)

    for iteration in range(first_iter, opt.iterations + 1):
        render_webui()
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        ending_refinement = iteration >= opt.densify_until_iter
        if iteration == opt.densify_until_iter:
            gaussians.set_lifetimes_to_max()

        # Loss
        def renderLosses(package, camera):
            image = package["render"]
            gt_image = camera.original_image.cuda()
            if camera.original_mask is not None:
                mask = camera.original_mask
                Ll1 = l1_loss(image * mask, gt_image * mask)
                Lssim = 1.0 - ssim(image * mask, gt_image * mask)
            else:
                Ll1 = l1_loss(image, gt_image)
                Lssim = 1.0 - ssim(image, gt_image)

            ### depth supervised loss
            depth = package["depth"]
            deploss = torch.zeros(1, device=Ll1.device)
            deploss_for_log = torch.zeros(1, device=Ll1.device)
            if usedepth and camera.original_depth is not None:
                depth_threshold = 1
                if apply_anchor_regularization and not ending_refinement:
                    depth_threshold = 0.2

                depth_mask = camera.original_depth > 0  # render_pkg["acc"][0]
                gt_maskeddepth = (camera.original_depth * depth_mask).cuda()
                if args.white_background:  # for 360 datasets ...
                    gt_maskeddepth = normalize_depth(gt_maskeddepth)
                    depth = normalize_depth(depth)

                deploss_for_log = l1_loss(gt_maskeddepth, depth * depth_mask) * 1.0
                if iteration > opt.random_depth_from_iter:
                    random_mask = torch.rand_like(gt_maskeddepth, device=depth_mask.device) < depth_threshold
                    deploss = l1_loss(gt_maskeddepth * random_mask, depth * depth_mask * random_mask) * 1.0
                else:
                    deploss = deploss_for_log
            return Ll1, Lssim, deploss, deploss_for_log

        Ll1, Lssim, deploss, deploss_for_log = renderLosses(render_pkg, viewpoint_cam)
        loss = (1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
        if use_projector and projector_camera == viewpoint_cam:
            loss = (
                opt.lambda_projector_image_loss * ((1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim)
                + opt.lambda_projector_depth_loss * deploss
            )
        else:
            loss = (1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim + deploss

        if use_fft_smooth and iteration > opt.fft_smooth_from_iter:
            # regularization with randomized camera
            random_camera.randomize()
            render_pkg = render(random_camera, gaussians, pipe, bg)
            random_image, random_visibility_filter, random_radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            gaussians.max_radii2D[random_visibility_filter] = torch.max(
                gaussians.max_radii2D[random_visibility_filter], random_radii[random_visibility_filter]
            )

            hf_kernel = get_fft_kernel(random_camera)

            # [c h w]
            image_weights = torch.fft.fft2(random_image)
            image_weights = torch.fft.fftshift(image_weights, dim=(-2, -1))
            image_weights = image_weights * hf_kernel
            image_weights = torch.fft.ifftshift(image_weights, dim=(-2, -1))
            image_weights = torch.abs(torch.fft.ifft2(image_weights))

            image_weights = image_weights.sum(dim=0, keepdim=True)
            gaussians.add_high_frequency_stats(random_camera, image_weights)

        if apply_anchor_regularization:
            lambda_ending_refinement_anchor = 1.2 if ending_refinement else 1
            anchor_loss = gaussians.get_anchor_loss()
            loss += anchor_loss["loss_anchor_geo"] * opt.lambda_anchor_geo * lambda_ending_refinement_anchor
            loss += anchor_loss["loss_anchor_color"] * opt.lambda_anchor_color
            if iteration % opt.opacity_reset_interval >= opt.densification_interval:  # not in opacity reset pass
                loss += anchor_loss["loss_anchor_opacity"] * opt.lambda_anchor_opacity
            loss += anchor_loss["loss_anchor_scale"] * opt.lambda_anchor_scale * lambda_ending_refinement_anchor

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_depthloss_for_log = 0.2 * deploss_for_log.item() + 0.8 * ema_depthloss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "Deploss": f"{ema_depthloss_for_log:.4f}",
                        "#pts": gaussians._xyz.shape[0],
                        "#hfpts": gaussians.hf_downgrade_count,
                    }
                )
                progress_bar.update(10)

                torch.cuda.empty_cache()

            if iteration % 100 == 0:
                # Enters anchor loss checking when depth overfits or not use depth
                if iteration > opt.anchor_min_iters and (not usedepth or ema_depthloss_for_log > prev_depthloss):
                    if use_anchor_loss and not apply_anchor_regularization:
                        print("[INFO] Hit depth overfitting bound, applying anchor regularization")
                        apply_anchor_regularization = True
                else:
                    prev_depthloss = ema_depthloss_for_log

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                txt_path=os.path.join(args.model_path, "metric.txt"),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.02 if apply_anchor_regularization else 1,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    while len(webgui.server.get_clients()) > 0:
        webgui.render_loop(gaussians, pipe, bg)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    txt_path=None,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    depth = render_pkg["depth"]

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    # [Taekkii-HARDCODING] Save images.
                    if txt_path is not None and config["name"] == "test":
                        elements = txt_path.split("/")
                        elements = elements[:-1]
                        render_path = os.path.join(*elements, f"render_{iteration:05d}")
                        os.makedirs(render_path, exist_ok=True)
                        render_image_path = os.path.join(render_path, f"{idx:03d}_render.png")
                        gt_image_path = os.path.join(render_path, f"{idx:03d}_gt.png")
                        torchvision.utils.save_image(image, render_image_path)
                        torchvision.utils.save_image(gt_image, gt_image_path)

                        # depth.
                        depth_path = os.path.join(render_path, f"{idx:03d}_depth.png")
                        depth = ((depth_colorize_with_mask(depth.cpu().numpy()[None])).squeeze() * 255.0).astype(
                            np.uint8
                        )
                        cv2.imwrite(depth_path, depth[:, :, ::-1])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                ssim_test /= len(config["cameras"])
                lpips_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.6f} PSNR {psnr_test:.2f}"
                )  # SSIM {ssim_test:.4f} LPIPS {lpips_test:.4f}")
                if config["name"] == "test":
                    with open(txt_path, "a") as fp:
                        print(f"{iteration}_{psnr_test:.6f}_{ssim_test:.6f}_{lpips_test:.6f}", file=fp)

                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - ssim", ssim_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - lpips", lpips_test, iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7000]
    )  # default=([1, 250, 500,]+ [i*1000 for i in range(1,31)]))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--use_projector", action="store_true")
    parser.add_argument("--use_fft_smooth", action="store_true")
    parser.add_argument("--use_anchor_loss", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        usedepth=args.depth,
        use_projector=args.use_projector,
        use_fft_smooth=args.use_fft_smooth,
        use_anchor_loss=args.use_anchor_loss,
    )

    # All done
    with open( os.path.join(args.model_path, "DONE.txt"), "w") as fp:
        print("DONE", file=fp)

    print("\nTraining complete.")
