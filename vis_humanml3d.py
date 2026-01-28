"""
Visualization script for HumanML3D motion data using viser.
Supports both skeleton visualization and SMPL mesh visualization.
"""
import time
from pathlib import Path
import viser
import viser.transforms as vtf
import tyro
import torch
import smplx
import numpy as np
from dataclasses import dataclass


# HumanML3D 22-joint skeleton definition
JOINT_NAMES = [
    'pelvis',      # 0
    'l_hip',       # 1
    'r_hip',       # 2
    'spine1',      # 3
    'l_knee',      # 4
    'r_knee',      # 5
    'spine2',      # 6
    'l_ankle',     # 7
    'r_ankle',     # 8
    'spine3',      # 9
    'l_foot',      # 10
    'r_foot',      # 11
    'neck',        # 12
    'l_collar',    # 13
    'r_collar',    # 14
    'head',        # 15
    'l_shoulder',  # 16
    'r_shoulder',  # 17
    'l_elbow',     # 18
    'r_elbow',     # 19
    'l_wrist',     # 20
    'r_wrist',     # 21
]

# Kinematic chain for skeleton visualization (parent -> child connections)
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],      # right leg
    [0, 1, 4, 7, 10],      # left leg
    [0, 3, 6, 9, 12, 15],  # spine to head
    [9, 14, 17, 19, 21],   # right arm
    [9, 13, 16, 18, 20],   # left arm
]

# Colors for each chain
CHAIN_COLORS = [
    (255, 100, 100),  # right leg - red
    (100, 100, 255),  # left leg - blue
    (100, 255, 100),  # spine - green
    (255, 200, 100),  # right arm - orange
    (100, 200, 255),  # left arm - cyan
]


@dataclass
class VisConfig:
    """Configuration for HumanML3D visualization."""
    motion_file: str = "./HumanML3D/new_joints/000000.npy"
    smpl_model_path: str = "/home/ubuntu/sky_workdir/egoallo/data"
    port: int = 8080
    show_mesh: bool = False  # Disabled by default - SMPL mesh needs proper conversion
    gender: str = "neutral"


def main(config: VisConfig):
    """Main visualization function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load motion data
    print(f"Loading motion from: {config.motion_file}")
    joints = np.load(config.motion_file)  # (T, 22, 3)
    num_frames = joints.shape[0]
    print(f"Loaded {num_frames} frames, shape: {joints.shape}")

    # Start viser server
    server = viser.ViserServer(port=config.port)
    server.gui.configure_theme(
        control_layout="collapsible",
        show_logo=False
    )

    # Camera reset button
    reset_camera = server.gui.add_button(
        label="Reset Up Direction",
        icon=viser.Icon.ARROW_BIG_UP_LINES,
        color="gray",
    )

    @reset_camera.on_click
    def _reset_camera_cb(_) -> None:
        for client in server.get_clients().values():
            client.camera.up_direction = vtf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

    # Playback controls
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=1, initial_value=20
        )

    # Visualization controls
    with server.gui.add_folder("Visualization"):
        gui_show_skeleton = server.gui.add_checkbox("Show Skeleton", True)
        gui_show_joints = server.gui.add_checkbox("Show Joint Points", True)
        gui_joint_size = server.gui.add_slider("Joint Size", min=0.01, max=0.1, step=0.005, initial_value=0.03)
        gui_bone_width = server.gui.add_slider("Bone Width", min=0.005, max=0.05, step=0.005, initial_value=0.015)
        gui_show_floor = server.gui.add_checkbox("Show Floor Grid", True)

    # Frame navigation callbacks
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Create floor grid
    floor_grid = server.scene.add_grid(
        "/floor",
        width=10,
        height=10,
        width_segments=20,
        height_segments=20,
        plane="xz",
        cell_color=(200, 200, 200),
        cell_thickness=1,
        visible=True,
    )

    # Create joint point cloud
    joint_cloud = server.scene.add_point_cloud(
        "/joints",
        points=joints[0],
        colors=np.array([[255, 255, 0]] * 22, dtype=np.uint8),
        point_size=gui_joint_size.value,
        visible=gui_show_joints.value,
    )

    # Create skeleton bones (line segments for each chain)
    bone_handles = []
    for chain_idx, chain in enumerate(KINEMATIC_CHAIN):
        color = CHAIN_COLORS[chain_idx]
        for i in range(len(chain) - 1):
            j1, j2 = chain[i], chain[i + 1]
            bone = server.scene.add_spline_catmull_rom(
                f"/skeleton/chain{chain_idx}/bone{i}",
                positions=np.array([joints[0, j1], joints[0, j2]]),
                color=color,
                line_width=gui_bone_width.value * 100,
                visible=gui_show_skeleton.value,
            )
            bone_handles.append((bone, j1, j2, chain_idx))

    def update_visualization(timestep: int):
        """Update all visualizations for the given timestep."""
        t = min(timestep, num_frames - 1)

        with server.atomic():
            # Update joint point cloud
            if gui_show_joints.value:
                joint_cloud.points = joints[t]
                joint_cloud.point_size = gui_joint_size.value
            joint_cloud.visible = gui_show_joints.value

            # Update skeleton bones
            for bone, j1, j2, _ in bone_handles:
                if gui_show_skeleton.value:
                    bone.positions = np.array([joints[t, j1], joints[t, j2]])
                    bone.line_width = gui_bone_width.value * 100
                bone.visible = gui_show_skeleton.value

            # Update floor visibility
            floor_grid.visible = gui_show_floor.value

    @gui_timestep.on_update
    def _(_) -> None:
        update_visualization(gui_timestep.value)

    @gui_show_skeleton.on_update
    def _(_) -> None:
        update_visualization(gui_timestep.value)

    @gui_show_joints.on_update
    def _(_) -> None:
        update_visualization(gui_timestep.value)

    @gui_joint_size.on_update
    def _(_) -> None:
        update_visualization(gui_timestep.value)

    @gui_bone_width.on_update
    def _(_) -> None:
        update_visualization(gui_timestep.value)

    @gui_show_floor.on_update
    def _(_) -> None:
        update_visualization(gui_timestep.value)

    # Initialize
    update_visualization(0)
    print(f"\nVisualization ready! Open http://localhost:{config.port} in your browser.")

    # Animation loop
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
