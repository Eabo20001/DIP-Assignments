"""Bundle adjustment for recovering a colored point cloud and camera poses.

The script optimizes:
- a shared focal length
- per-view camera rotation (Euler XYZ) and translation
- 3D coordinates for all points

Outputs are written to the chosen output directory:
- loss_curve.png
- reconstruction.obj
- point_cloud_preview.png
- metrics.json
- points3d.npy
- camera_parameters.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGE_SIZE = 1024
IMAGE_CENTER = IMAGE_SIZE / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch bundle adjustment")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "bundle_adjustment",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--init-distance", type=float, default=2.5)
    parser.add_argument("--init-fov-deg", type=float, default=60.0)
    parser.add_argument("--yaw-range-deg", type=float, default=70.0)
    parser.add_argument("--lr-points", type=float, default=5e-2)
    parser.add_argument("--lr-poses", type=float, default=5e-3)
    parser.add_argument("--lr-focal", type=float, default=5e-2)
    parser.add_argument("--point-reg-weight", type=float, default=1e-4)
    parser.add_argument("--point-center-weight", type=float, default=1e-2)
    parser.add_argument("--pose-anchor-weight", type=float, default=1e-3)
    parser.add_argument("--focal-anchor-weight", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    threshold = torch.tensor(20.0, dtype=x.dtype, device=x.device)
    return torch.where(x > threshold, x, torch.log(torch.expm1(x)))


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    points2d = np.load(data_dir / "points2d.npz")
    view_keys = sorted(points2d.files)
    observations = np.stack([points2d[key] for key in view_keys], axis=0).astype(np.float32)
    colors = np.load(data_dir / "points3d_colors.npy").astype(np.float32)
    return observations, colors, view_keys


def focal_from_fov(image_size: int, fov_deg: float) -> float:
    half_fov = math.radians(fov_deg) / 2.0
    return 0.5 * image_size / math.tan(half_fov)


def rotation_matrix_x(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    row0 = torch.stack([one, zero, zero], dim=-1)
    row1 = torch.stack([zero, c, -s], dim=-1)
    row2 = torch.stack([zero, s, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rotation_matrix_y(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    row0 = torch.stack([c, zero, s], dim=-1)
    row1 = torch.stack([zero, one, zero], dim=-1)
    row2 = torch.stack([-s, zero, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def rotation_matrix_z(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    row0 = torch.stack([c, -s, zero], dim=-1)
    row1 = torch.stack([s, c, zero], dim=-1)
    row2 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def euler_xyz_to_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    rx = rotation_matrix_x(euler_angles[..., 0])
    ry = rotation_matrix_y(euler_angles[..., 1])
    rz = rotation_matrix_z(euler_angles[..., 2])
    return rx @ ry @ rz


def initial_camera_guess(
    num_views: int,
    init_distance: float,
    yaw_range_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    euler = np.zeros((num_views, 3), dtype=np.float32)
    euler[:, 1] = np.linspace(
        -math.radians(yaw_range_deg),
        math.radians(yaw_range_deg),
        num_views,
        dtype=np.float32,
    )
    translations = np.zeros((num_views, 3), dtype=np.float32)
    translations[:, 2] = -init_distance
    return euler, translations


def euler_xyz_to_matrix_np(euler_angles: np.ndarray) -> np.ndarray:
    angles = torch.from_numpy(euler_angles)
    return euler_xyz_to_matrix(angles).numpy()


def triangulate_points(
    observations: np.ndarray,
    mask: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    rotations_inv = np.transpose(rotations, (0, 2, 1))
    camera_centers = -np.einsum("vij,vj->vi", rotations_inv, translations)

    u = observations[..., 0]
    v = observations[..., 1]
    dirs_cam = np.stack(
        [
            (u - cx) / focal,
            -(v - cy) / focal,
            -np.ones_like(u),
        ],
        axis=-1,
    )
    dirs_world = np.einsum("vij,vnj->vni", rotations_inv, dirs_cam)
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True).clip(min=1e-8)

    eye = np.eye(3, dtype=np.float32)[None, None]
    line_projectors = eye - dirs_world[..., :, None] * dirs_world[..., None, :]
    line_projectors *= mask[..., None, None]

    lhs = line_projectors.sum(axis=0)
    rhs = np.einsum("vnij,vj->vni", line_projectors, camera_centers).sum(axis=0)

    visibility_count = mask.sum(axis=0)
    reg_strength = np.where(visibility_count >= 2, 1e-4, 1.0).astype(np.float32)
    lhs += reg_strength[:, None, None] * np.eye(3, dtype=np.float32)[None]

    points = np.linalg.solve(lhs, rhs[..., None]).squeeze(-1)
    points -= points.mean(axis=0, keepdims=True)
    return points.astype(np.float32)


class BundleAdjustmentModel(nn.Module):
    def __init__(
        self,
        init_points: torch.Tensor,
        init_euler: torch.Tensor,
        init_translations: torch.Tensor,
        init_focal: float,
        min_focal: float = 50.0,
        min_distance: float = 0.25,
    ) -> None:
        super().__init__()
        self.points = nn.Parameter(init_points.clone())
        self.euler = nn.Parameter(init_euler.clone())
        self.translation_xy = nn.Parameter(init_translations[:, :2].clone())

        translation_depth = (-init_translations[:, 2] - min_distance).clamp_min(1e-3)
        self.translation_depth_raw = nn.Parameter(inverse_softplus(translation_depth))

        focal_target = torch.tensor(
            max(init_focal - min_focal, 1e-3),
            dtype=init_points.dtype,
            device=init_points.device,
        )
        self.focal_raw = nn.Parameter(inverse_softplus(focal_target))

        self.min_focal = float(min_focal)
        self.min_distance = float(min_distance)

        self.register_buffer("init_euler", init_euler.clone())
        self.register_buffer("init_translations", init_translations.clone())
        self.register_buffer(
            "init_focal_tensor",
            torch.tensor(init_focal, dtype=init_points.dtype, device=init_points.device),
        )

    @property
    def focal(self) -> torch.Tensor:
        return F.softplus(self.focal_raw) + self.min_focal

    @property
    def translations(self) -> torch.Tensor:
        depth = F.softplus(self.translation_depth_raw) + self.min_distance
        z = -depth.unsqueeze(-1)
        return torch.cat([self.translation_xy, z], dim=-1)

    @property
    def rotations(self) -> torch.Tensor:
        return euler_xyz_to_matrix(self.euler)

    def project(self, cx: float, cy: float) -> torch.Tensor:
        camera_points = torch.einsum("vij,nj->vni", self.rotations, self.points)
        camera_points = camera_points + self.translations[:, None, :]

        z = camera_points[..., 2]
        z_sign = torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
        safe_z = z_sign * z.abs().clamp_min(1e-4)

        u = -self.focal * camera_points[..., 0] / safe_z + cx
        v = self.focal * camera_points[..., 1] / safe_z + cy
        return torch.stack([u, v], dim=-1)


def compute_losses(
    model: BundleAdjustmentModel,
    observations: torch.Tensor,
    visibility_mask: torch.Tensor,
    cx: float,
    cy: float,
    point_reg_weight: float,
    point_center_weight: float,
    pose_anchor_weight: float,
    focal_anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    projected = model.project(cx, cy)
    residuals = projected - observations[..., :2]
    squared_error = residuals.square().sum(dim=-1)
    reprojection_loss = squared_error[visibility_mask].mean()

    point_reg = model.points.square().mean()
    point_center = model.points.mean(dim=0).square().sum()
    pose_anchor = (
        (model.euler - model.init_euler).square().mean()
        + (model.translations - model.init_translations).square().mean()
    )
    focal_anchor = ((model.focal - model.init_focal_tensor) / model.init_focal_tensor).square()

    total_loss = (
        reprojection_loss
        + point_reg_weight * point_reg
        + point_center_weight * point_center
        + pose_anchor_weight * pose_anchor
        + focal_anchor_weight * focal_anchor
    )

    metrics = {
        "total_loss": total_loss,
        "reprojection_loss": reprojection_loss,
        "rmse_px": torch.sqrt(reprojection_loss),
        "point_reg": point_reg,
        "point_center": point_center,
        "pose_anchor": pose_anchor,
        "focal_anchor": focal_anchor,
        "focal": model.focal.detach(),
    }
    return total_loss, metrics


def clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def save_loss_curve(loss_history: list[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(loss_history, color="#005f73", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("Bundle Adjustment Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_point_cloud_preview(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
    max_points: int = 10000,
) -> None:
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
        colors = colors[indices]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1.5)
    ax.view_init(elev=12, azim=-55)
    ax.set_title("Recovered 3D Point Cloud")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.5 * (maxs - mins).max()
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def save_obj(points: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for point, color in zip(points, colors):
            handle.write(
                f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    observations_np, colors_np, view_keys = load_dataset(args.data_dir)
    visibility_mask_np = observations_np[..., 2] > 0.5

    init_focal = focal_from_fov(IMAGE_SIZE, args.init_fov_deg)
    init_euler_np, init_translations_np = initial_camera_guess(
        num_views=len(view_keys),
        init_distance=args.init_distance,
        yaw_range_deg=args.yaw_range_deg,
    )
    init_rotations_np = euler_xyz_to_matrix_np(init_euler_np)
    init_points_np = triangulate_points(
        observations=observations_np[..., :2],
        mask=visibility_mask_np.astype(np.float32),
        rotations=init_rotations_np,
        translations=init_translations_np,
        focal=init_focal,
        cx=IMAGE_CENTER,
        cy=IMAGE_CENTER,
    )

    observations = torch.from_numpy(observations_np).to(device=device, dtype=torch.float32)
    visibility_mask = torch.from_numpy(visibility_mask_np).to(device=device, dtype=torch.bool)

    model = BundleAdjustmentModel(
        init_points=torch.from_numpy(init_points_np).to(device=device, dtype=torch.float32),
        init_euler=torch.from_numpy(init_euler_np).to(device=device, dtype=torch.float32),
        init_translations=torch.from_numpy(init_translations_np).to(device=device, dtype=torch.float32),
        init_focal=init_focal,
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": [model.points], "lr": args.lr_points},
            {
                "params": [
                    model.euler,
                    model.translation_xy,
                    model.translation_depth_raw,
                ],
                "lr": args.lr_poses,
            },
            {"params": [model.focal_raw], "lr": args.lr_focal},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.iterations,
        eta_min=args.lr_poses * 0.1,
    )

    loss_history: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    start_time = perf_counter()
    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)

        total_loss, metrics = compute_losses(
            model=model,
            observations=observations,
            visibility_mask=visibility_mask,
            cx=IMAGE_CENTER,
            cy=IMAGE_CENTER,
            point_reg_weight=args.point_reg_weight,
            point_center_weight=args.point_center_weight,
            pose_anchor_weight=args.pose_anchor_weight,
            focal_anchor_weight=args.focal_anchor_weight,
        )
        total_loss.backward()

        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        optimizer.step()
        scheduler.step()

        current_loss = float(metrics["total_loss"].detach().cpu())
        loss_history.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            best_state = clone_state_dict(model)

        if iteration == 1 or iteration % args.print_every == 0 or iteration == args.iterations:
            elapsed = perf_counter() - start_time
            print(
                f"[{iteration:04d}/{args.iterations}] "
                f"loss={current_loss:.6f} "
                f"reproj={float(metrics['reprojection_loss'].detach().cpu()):.6f} "
                f"rmse={float(metrics['rmse_px'].detach().cpu()):.4f}px "
                f"f={float(metrics['focal'].detach().cpu()):.3f} "
                f"time={elapsed:.1f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        _, final_metrics = compute_losses(
            model=model,
            observations=observations,
            visibility_mask=visibility_mask,
            cx=IMAGE_CENTER,
            cy=IMAGE_CENTER,
            point_reg_weight=args.point_reg_weight,
            point_center_weight=args.point_center_weight,
            pose_anchor_weight=args.pose_anchor_weight,
            focal_anchor_weight=args.focal_anchor_weight,
        )
        points_final = model.points.detach().cpu().numpy()
        euler_final = model.euler.detach().cpu().numpy()
        rotations_final = model.rotations.detach().cpu().numpy()
        translations_final = model.translations.detach().cpu().numpy()
        focal_final = float(model.focal.detach().cpu())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_loss_curve(loss_history, args.output_dir / "loss_curve.png")
    save_point_cloud_preview(points_final, colors_np, args.output_dir / "point_cloud_preview.png")
    save_obj(points_final, colors_np, args.output_dir / "reconstruction.obj")
    np.save(args.output_dir / "points3d.npy", points_final)
    np.savez(
        args.output_dir / "camera_parameters.npz",
        view_keys=np.array(view_keys),
        euler_rad=euler_final,
        rotations=rotations_final,
        translations=translations_final,
        focal_length=np.array([focal_final], dtype=np.float32),
    )

    metrics_payload = {
        "device": str(device),
        "iterations": args.iterations,
        "num_views": int(observations_np.shape[0]),
        "num_points": int(observations_np.shape[1]),
        "final_total_loss": float(final_metrics["total_loss"].detach().cpu()),
        "final_reprojection_loss": float(final_metrics["reprojection_loss"].detach().cpu()),
        "final_rmse_px": float(final_metrics["rmse_px"].detach().cpu()),
        "focal_length": focal_final,
        "mean_translation": translations_final.mean(axis=0).tolist(),
        "mean_euler_rad": euler_final.mean(axis=0).tolist(),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    np.save(args.output_dir / "loss_history.npy", np.array(loss_history, dtype=np.float32))

    print(f"Saved results to: {args.output_dir.resolve()}")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
