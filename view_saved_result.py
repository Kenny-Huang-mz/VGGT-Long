import argparse
import time

import numpy as np
import viser
import viser.transforms as viser_tf


def compute_fov(intrinsic, height):
    fy = intrinsic[1, 1]
    if fy <= 0:
        fy = 1.1 * height
    return 2 * np.arctan2(height / 2, fy)


def main():
    parser = argparse.ArgumentParser(description="View a saved VGGT-Long reconstruction .npz with viser.")
    parser.add_argument("--npz_path", type=str, required=True, help="Saved .npz path.")
    parser.add_argument("--port", type=int, default=6008, help="Viser port.")
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=85.0,
        help="Initial confidence percentile to keep.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.001,
        help="Rendered point size.",
    )
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=False)
    images = data["images"]
    world_points = data["world_points"]
    world_points_conf = (
        data["world_points_conf"]
        if "world_points_conf" in data.files
        else data["conf_prob"]
        if "conf_prob" in data.files
        else 1.0 / (1.0 + np.exp(-data["conf"]))
    )
    camera_poses = data["camera_poses"]
    intrinsics = data["intrinsic"] if "intrinsic" in data.files else data["intrinsics"]
    image_names = data["image_names"] if "image_names" in data.files else np.arange(images.shape[0]).astype(str)

    n_frames, _, height, width = images.shape
    points = world_points.reshape(-1, 3)
    colors = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = world_points_conf.reshape(-1)
    frame_indices = np.repeat(np.arange(n_frames), height * width)

    valid_points = np.isfinite(points).all(axis=1)
    scene_center = points[valid_points].mean(axis=0) if np.any(valid_points) else np.zeros(3, dtype=np.float32)
    points_centered = points - scene_center
    camera_poses = camera_poses.copy()
    camera_poses[:, :3, 3] -= scene_center

    print(f"Loaded: {args.npz_path}")
    print(f"Frames: {n_frames}, image size: {width}x{height}")
    print(f"Starting viser server on port {args.port}")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent",
        min=0.0,
        max=100.0,
        step=0.1,
        initial_value=args.conf_threshold,
    )
    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames",
        options=["All"] + [str(i) for i in range(n_frames)],
        initial_value="All",
    )

    init_threshold_val = np.percentile(conf_flat, args.conf_threshold)
    init_mask = valid_points & (conf_flat >= init_threshold_val) & (conf_flat > 1e-5)
    point_cloud = server.scene.add_point_cloud(
        name="vggt_long_pcd",
        points=points_centered[init_mask],
        colors=colors[init_mask],
        point_size=args.point_size,
        point_shape="circle",
    )

    frames = []
    frustums = []

    def visualize_frames():
        for frame in frames:
            frame.remove()
        frames.clear()
        for frustum in frustums:
            frustum.remove()
        frustums.clear()

        for idx in range(n_frames):
            cam_to_world = camera_poses[idx]
            frame_pose = viser_tf.SE3.from_matrix(cam_to_world)
            frame = server.scene.add_frame(
                f"frame_{idx}",
                wxyz=frame_pose.rotation().wxyz,
                position=frame_pose.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame)

            image = (images[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
            fov = compute_fov(intrinsics[idx], height)
            frustum = server.scene.add_camera_frustum(
                f"frame_{idx}/frustum",
                fov=fov,
                aspect=width / height,
                scale=0.05,
                image=image,
                line_width=1.0,
            )
            frustums.append(frustum)

            @frustum.on_click
            def _(_, frame_handle=frame):
                for client in server.get_clients().values():
                    client.camera.wxyz = frame_handle.wxyz
                    client.camera.position = frame_handle.position

    def update_point_cloud():
        threshold_val = np.percentile(conf_flat, gui_points_conf.value)
        conf_mask = valid_points & (conf_flat >= threshold_val) & (conf_flat > 1e-5)
        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            frame_mask = frame_indices == int(gui_frame_selector.value)
        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors[combined_mask]

    @gui_points_conf.on_update
    def _(_event):
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_event):
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_event):
        for frame in frames:
            frame.visible = gui_show_frames.value
        for frustum in frustums:
            frustum.visible = gui_show_frames.value

    visualize_frames()
    print("Viewer started.")
    while True:
        time.sleep(0.01)


if __name__ == "__main__":
    main()
