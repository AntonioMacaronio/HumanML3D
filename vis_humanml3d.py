"""
Visualization script for HumanML3D motion data using viser.
Supports both skeleton visualization (from new_joints/) and SMPL mesh visualization (from smpl/).

Usage:
    # Visualize skeleton from new_joints/
    python vis_humanml3d.py --motion_id 000000

    # Visualize SMPL mesh from smpl/
    python vis_humanml3d.py --motion_id 000000 --show_mesh
"""
import os
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
    motion_id: str = "000000"
    smpl_model_path: str = "./body_models"
    joints_dir: str = "./HumanML3D/new_joints"
    smpl_dir: str = "./HumanML3D/smpl"
    texts_dir: str = "./HumanML3D/texts"
    port: int = 8080
    show_mesh: bool = True
    gender: str = "neutral"


def load_text_descriptions(texts_dir: str, motion_id: str) -> list[str]:
    """Load text descriptions for a motion."""
    # Remove 'M' prefix if present (mirrored motions use same text)
    text_id = motion_id.lstrip('M')
    path = os.path.join(texts_dir, f"{text_id}.txt")
    if not os.path.exists(path):
        return ["No text description available"]

    descriptions = []
    with open(path, 'r') as f:
        for line in f:
            # Format: description#POS_tagged#start#end
            parts = line.strip().split('#')
            if parts:
                descriptions.append(parts[0])
    return descriptions


def compute_smpl_vertices(
    smpl_model,
    bdata_poses: np.ndarray,
    bdata_trans: np.ndarray,
    betas: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Convert SMPL-H poses to SMPL vertices.

    Args:
        bdata_poses: (num_frames, 156) SMPL-H axis-angle poses
        bdata_trans: (num_frames, 3) translations
        betas: (10,) or (16,) shape parameters

    Returns:
        vertices: (num_frames, 6890, 3) numpy array
    """
    num_frames = bdata_poses.shape[0]

    # SMPL model expects:
    # - global_orient: (B, 3) axis-angle for root
    # - body_pose: (B, 69) axis-angle for 23 body joints
    # HumanML3D stores SMPL-H format (52 joints * 3 = 156), we use first 72 values
    global_orient = torch.tensor(bdata_poses[:, :3], dtype=torch.float32, device=device)
    body_pose = torch.tensor(bdata_poses[:, 3:72], dtype=torch.float32, device=device)
    transl = torch.tensor(bdata_trans, dtype=torch.float32, device=device)
    betas_tensor = torch.tensor(betas[:10], dtype=torch.float32, device=device).unsqueeze(0).expand(num_frames, -1)

    vertices_list = []
    batch_size = 64  # Process in batches to avoid OOM

    for i in range(0, num_frames, batch_size):
        end_idx = min(i + batch_size, num_frames)

        with torch.no_grad():
            output = smpl_model(
                global_orient=global_orient[i:end_idx],
                body_pose=body_pose[i:end_idx],
                transl=transl[i:end_idx],
                betas=betas_tensor[i:end_idx],
            )
        vertices_list.append(output.vertices.cpu().numpy())

    return np.concatenate(vertices_list, axis=0)


def main(config: VisConfig):
    """Main visualization function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load text descriptions
    texts = load_text_descriptions(config.texts_dir, config.motion_id)
    print(f"\nMotion ID: {config.motion_id}")
    print(f"Text descriptions ({len(texts)}):")
    for i, text in enumerate(texts[:4]):
        print(f"  {i+1}. {text}")

    # Load motion data
    joints = None
    vertices = None
    smpl_model = None
    num_joints_frames = 0
    num_mesh_frames = 0

    # Load SMPL mesh data if requested
    if config.show_mesh:
        smpl_path = os.path.join(config.smpl_dir, f"{config.motion_id}.npy")
        if os.path.exists(smpl_path):
            print(f"\nLoading SMPL data from: {smpl_path}")
            smpl_data = np.load(smpl_path, allow_pickle=True).item()

            bdata_poses = smpl_data['bdata_poses']
            bdata_trans = smpl_data['bdata_trans']
            betas = smpl_data['betas']
            gender = smpl_data.get('gender', 'neutral')

            num_mesh_frames = bdata_poses.shape[0]
            print(f"  Frames: {num_mesh_frames}")
            print(f"  Gender: {gender}")
            print(f"  Poses shape: {bdata_poses.shape}")

            # Load SMPL model
            print("Loading SMPL model...")
            gender_map = {'male': 'male', 'female': 'female', 'neutral': 'neutral', 'unknown': 'neutral'}
            smpl_gender = gender_map.get(gender.item(), 'neutral')

            smpl_model = smplx.create(
                model_path=config.smpl_model_path,
                model_type='smpl',
                gender=smpl_gender,
                num_betas=10,
            ).to(device)

            # Compute vertices
            print("Computing SMPL vertices...")
            vertices = compute_smpl_vertices(smpl_model, bdata_poses, bdata_trans, betas, device)
            print(f"  Vertices shape: {vertices.shape}")

            # Also get joints from SMPL data if not loaded from new_joints
            if joints is None:
                joints = smpl_data.get('jtr', None)
                if joints is not None:
                    print(f"  Using joints from SMPL data: {joints.shape}")
        else:
            print(f"Warning: SMPL data not found at {smpl_path}")

    if joints is None and vertices is None:
        raise FileNotFoundError(f"No motion data found for {config.motion_id}")

    # Determine num_frames as the minimum of available data sources
    frame_counts = []
    if joints is not None:
        frame_counts.append(joints.shape[0])
    if vertices is not None:
        frame_counts.append(vertices.shape[0])
    num_frames = min(frame_counts)
    print(f"\nUsing {num_frames} frames (min of available sources)")

    # Start viser server
    print(f"\nStarting viser server on port {config.port}...")
    server = viser.ViserServer(port=config.port)
    server.gui.configure_theme(control_layout="collapsible", show_logo=False)

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

    # Motion info panel
    with server.gui.add_folder("Motion Info"):
        server.gui.add_markdown(f"**Motion ID:** {config.motion_id}")
        server.gui.add_markdown(f"**Frames:** {num_frames} ({num_frames / 20.0:.1f}s at 20 FPS)")
        server.gui.add_markdown("---")
        server.gui.add_markdown("**Text Descriptions:**")
        for i, text in enumerate(texts[:4]):
            server.gui.add_markdown(f"{i+1}. _{text}_")

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
        gui_loop = server.gui.add_checkbox("Loop", True)

    # Visualization controls
    with server.gui.add_folder("Visualization"):
        gui_show_skeleton = server.gui.add_checkbox("Show Skeleton", joints is not None)
        gui_show_joints = server.gui.add_checkbox("Show Joint Points", joints is not None)
        gui_show_mesh = server.gui.add_checkbox("Show SMPL Mesh", vertices is not None)
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
        plane="xy",
        cell_color=(200, 200, 200),
        cell_thickness=1,
        visible=True,
    )

    # Create joint point cloud (if joints available)
    joint_cloud = None
    bone_handles = []
    if joints is not None:
        joint_cloud = server.scene.add_point_cloud(
            "/joints",
            points=joints[0],
            colors=np.array([[255, 255, 0]] * joints.shape[1], dtype=np.uint8),
            point_size=gui_joint_size.value,
            visible=gui_show_joints.value,
        )

        # Create skeleton bones
        for chain_idx, chain in enumerate(KINEMATIC_CHAIN):
            color = CHAIN_COLORS[chain_idx]
            for i in range(len(chain) - 1):
                j1, j2 = chain[i], chain[i + 1]
                if j1 < joints.shape[1] and j2 < joints.shape[1]:
                    bone = server.scene.add_spline_catmull_rom(
                        f"/skeleton/chain{chain_idx}/bone{i}",
                        positions=np.array([joints[0, j1], joints[0, j2]]),
                        color=color,
                        line_width=gui_bone_width.value * 100,
                        visible=gui_show_skeleton.value,
                    )
                    bone_handles.append((bone, j1, j2, chain_idx))

    # Create SMPL mesh (if vertices available)
    mesh_handle = None
    if vertices is not None and smpl_model is not None:
        mesh_handle = server.scene.add_mesh_simple(
            "/smpl_mesh",
            vertices=vertices[0],
            faces=smpl_model.faces,
            color=(100, 149, 237),  # Cornflower blue
            wireframe=False,
            visible=gui_show_mesh.value,
        )

    def update_visualization(timestep: int):
        """Update all visualizations for the given timestep."""
        t = min(timestep, num_frames - 1)

        with server.atomic():
            # Update joint point cloud
            if joint_cloud is not None:
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

            # Update SMPL mesh
            if mesh_handle is not None:
                if gui_show_mesh.value:
                    mesh_handle.vertices = vertices[t]
                mesh_handle.visible = gui_show_mesh.value

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

    @gui_show_mesh.on_update
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
            next_frame = gui_timestep.value + 1
            if next_frame >= num_frames:
                if gui_loop.value:
                    next_frame = 0
                else:
                    gui_playing.value = False
                    next_frame = num_frames - 1
            gui_timestep.value = next_frame

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
