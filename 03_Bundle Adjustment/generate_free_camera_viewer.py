"""Gradio viewer for interactively orbiting around a reconstructed point cloud.

Run with:
    conda run -n DIP python generate_free_camera_viewer.py
"""

from __future__ import annotations

import argparse
import html
import json
import socket
from pathlib import Path

import gradio as gr
import numpy as np


DEFAULT_POINTS_PATH = Path("outputs/bundle_adjustment/points3d.npy")
DEFAULT_OBJ_PATH = Path("outputs/bundle_adjustment/reconstruction.obj")
DEFAULT_COLORS_PATH = Path("data/points3d_colors.npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive free-camera point-cloud viewer")
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def load_obj_vertices(obj_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    points: list[list[float]] = []
    colors: list[list[float]] = []

    with obj_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(parts) >= 7:
                colors.append([float(parts[4]), float(parts[5]), float(parts[6])])

    if not points:
        raise ValueError(f"No vertex records were found in {obj_path}.")

    color_array = np.asarray(colors, dtype=np.float32) if colors else None
    return np.asarray(points, dtype=np.float32), color_array


def load_point_cloud(points_path: str, colors_path: str) -> tuple[np.ndarray, np.ndarray]:
    point_file = Path(points_path).expanduser()
    color_file = Path(colors_path).expanduser()

    if not point_file.exists():
        raise FileNotFoundError(
            f"Point-cloud file not found: {point_file}. "
            "Run bundle_adjustment.py first, or point the viewer at an existing .npy or .obj file."
        )

    if point_file.suffix.lower() == ".npy":
        points = np.load(point_file).astype(np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected Nx3 points in {point_file}, but found shape {points.shape}.")

        if color_file.exists():
            colors = np.load(color_file).astype(np.float32)
            if colors.shape != points.shape:
                raise ValueError(
                    f"Color array shape {colors.shape} does not match point-cloud shape {points.shape}."
                )
        else:
            colors = np.full_like(points, 0.85, dtype=np.float32)
    elif point_file.suffix.lower() == ".obj":
        points, obj_colors = load_obj_vertices(point_file)
        if obj_colors is not None and obj_colors.shape == points.shape:
            colors = obj_colors
        elif color_file.exists():
            colors = np.load(color_file).astype(np.float32)
            if colors.shape != points.shape:
                raise ValueError(
                    f"Color array shape {colors.shape} does not match OBJ point-cloud shape {points.shape}."
                )
        else:
            colors = np.full_like(points, 0.85, dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported point-cloud format: {point_file.suffix}. Use a .npy array or an OBJ with vertex lines."
        )

    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)
    return points, colors


def sample_point_cloud(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors

    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices], colors[indices]


def normalize_points(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1).max()
    if radius < 1e-8:
        return centered
    return centered / radius


def build_viewer_srcdoc(
    points: np.ndarray,
    colors: np.ndarray,
    point_size: float,
    background: str,
) -> str:
    point_payload = json.dumps(points.tolist(), separators=(",", ":"))
    color_payload = json.dumps(colors.tolist(), separators=(",", ":"))
    point_size = float(point_size)
    background = background.strip() or "#101820"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {{
      color-scheme: dark;
      --panel-bg: rgba(8, 12, 16, 0.72);
      --panel-border: rgba(255, 255, 255, 0.14);
      --text-main: #f4f7fb;
      --text-muted: rgba(244, 247, 251, 0.78);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      background: {background};
      overflow: hidden;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }}
    #canvas {{
      display: block;
      width: 100vw;
      height: 100vh;
      cursor: grab;
    }}
    #canvas.dragging {{
      cursor: grabbing;
    }}
    .panel {{
      position: fixed;
      left: 18px;
      top: 18px;
      z-index: 10;
      max-width: 320px;
      padding: 14px 16px;
      border-radius: 16px;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      backdrop-filter: blur(12px);
      color: var(--text-main);
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
    }}
    .panel h1 {{
      margin: 0 0 8px;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0.01em;
    }}
    .panel p {{
      margin: 0;
      color: var(--text-muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .stats {{
      margin-top: 10px;
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 6px 10px;
      font-size: 12px;
      color: var(--text-muted);
    }}
    .stats strong {{
      color: var(--text-main);
      font-weight: 600;
    }}
    .actions {{
      margin-top: 12px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      color: var(--text-main);
      padding: 7px 12px;
      font-size: 12px;
      cursor: pointer;
    }}
    button:hover {{
      background: rgba(255, 255, 255, 0.12);
    }}
  </style>
</head>
<body>
  <canvas id="canvas"></canvas>
  <section class="panel">
    <h1>Free Camera Viewer</h1>
    <p>Drag to orbit, use the mouse wheel to zoom, and double-click to reset the view.</p>
    <div class="stats">
      <span>Points</span><strong id="point-count"></strong>
      <span>Yaw</span><strong id="yaw-value"></strong>
      <span>Pitch</span><strong id="pitch-value"></strong>
      <span>Zoom</span><strong id="zoom-value"></strong>
    </div>
    <div class="actions">
      <button id="reset-view" type="button">Reset View</button>
      <button id="fit-view" type="button">Fit To Screen</button>
    </div>
  </section>
  <script>
    const points = {point_payload};
    const colors = {color_payload};
    const pointSize = {point_size:.3f};

    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const pointCountLabel = document.getElementById("point-count");
    const yawValue = document.getElementById("yaw-value");
    const pitchValue = document.getElementById("pitch-value");
    const zoomValue = document.getElementById("zoom-value");
    const resetButton = document.getElementById("reset-view");
    const fitButton = document.getElementById("fit-view");

    let width = 0;
    let height = 0;
    let pixelRatio = 1;
    let yaw = -0.5;
    let pitch = 0.2;
    let radius = 3.0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let needsRender = true;

    pointCountLabel.textContent = String(points.length);

    function clamp(value, minValue, maxValue) {{
      return Math.min(maxValue, Math.max(minValue, value));
    }}

    function resizeCanvas() {{
      pixelRatio = Math.max(1, window.devicePixelRatio || 1);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * pixelRatio);
      canvas.height = Math.floor(height * pixelRatio);
      canvas.style.width = `${{width}}px`;
      canvas.style.height = `${{height}}px`;
      ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      needsRender = true;
    }}

    function resetView() {{
      yaw = -0.5;
      pitch = 0.2;
      radius = 3.0;
      needsRender = true;
    }}

    function fitView() {{
      radius = 2.8;
      needsRender = true;
    }}

    function rotatePoint(point) {{
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const cp = Math.cos(pitch);
      const sp = Math.sin(pitch);

      const x1 = cy * point[0] + sy * point[2];
      const z1 = -sy * point[0] + cy * point[2];

      const y2 = cp * point[1] - sp * z1;
      const z2 = sp * point[1] + cp * z1;

      return [x1, y2, z2];
    }}

    function render() {{
      if (!needsRender) {{
        requestAnimationFrame(render);
        return;
      }}

      needsRender = false;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "{background}";
      ctx.fillRect(0, 0, width, height);

      const focal = Math.min(width, height) * 0.8;
      const projected = [];
      for (let i = 0; i < points.length; i += 1) {{
        const rotated = rotatePoint(points[i]);
        const depth = rotated[2] + radius;
        if (depth <= 0.05) {{
          continue;
        }}
        const scale = focal / depth;
        projected.push({{
          x: width * 0.5 + rotated[0] * scale,
          y: height * 0.5 - rotated[1] * scale,
          depth,
          color: colors[i],
        }});
      }}

      projected.sort((a, b) => b.depth - a.depth);
      for (const item of projected) {{
        const alpha = clamp(1.35 - item.depth / (radius + 1.2), 0.16, 1.0);
        const size = clamp(pointSize * (focal / item.depth) * 0.018, 0.8, pointSize * 3.2);
        ctx.beginPath();
        ctx.fillStyle = `rgba(${{Math.round(item.color[0] * 255)}}, ${{Math.round(item.color[1] * 255)}}, ${{Math.round(item.color[2] * 255)}}, ${{alpha.toFixed(4)}})`;
        ctx.arc(item.x, item.y, size, 0, Math.PI * 2);
        ctx.fill();
      }}

      yawValue.textContent = `${{(yaw * 180 / Math.PI).toFixed(1)}}°`;
      pitchValue.textContent = `${{(pitch * 180 / Math.PI).toFixed(1)}}°`;
      zoomValue.textContent = radius.toFixed(2);

      requestAnimationFrame(render);
    }}

    canvas.addEventListener("mousedown", (event) => {{
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.classList.add("dragging");
    }});

    window.addEventListener("mouseup", () => {{
      dragging = false;
      canvas.classList.remove("dragging");
    }});

    window.addEventListener("mousemove", (event) => {{
      if (!dragging) {{
        return;
      }}
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      lastX = event.clientX;
      lastY = event.clientY;

      yaw += dx * 0.008;
      pitch = clamp(pitch + dy * 0.008, -1.4, 1.4);
      needsRender = true;
    }});

    canvas.addEventListener("wheel", (event) => {{
      event.preventDefault();
      radius = clamp(radius + event.deltaY * 0.003, 1.2, 9.0);
      needsRender = true;
    }}, {{ passive: false }});

    canvas.addEventListener("dblclick", () => {{
      resetView();
    }});

    window.addEventListener("resize", resizeCanvas);
    resetButton.addEventListener("click", resetView);
    fitButton.addEventListener("click", fitView);

    resizeCanvas();
    resetView();
    requestAnimationFrame(render);
  </script>
</body>
</html>"""


def build_iframe_html(srcdoc: str) -> str:
    escaped_srcdoc = html.escape(srcdoc, quote=True)
    return (
        '<iframe '
        'style="width: 100%; height: 760px; border: 0; border-radius: 18px; overflow: hidden; '
        'box-shadow: 0 18px 44px rgba(0, 0, 0, 0.18);" '
        f'srcdoc="{escaped_srcdoc}"></iframe>'
    )


def build_message_card(message: str) -> str:
    return f"""
    <div style="
        min-height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 18px;
        padding: 32px;
        background: linear-gradient(135deg, #0f1720, #16202a);
        color: #f8fafc;
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.16);
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        text-align: center;
        line-height: 1.6;
    ">
      <div style="max-width: 700px; font-size: 15px;">{html.escape(message)}</div>
    </div>
    """


def build_viewer(
    points_path: str,
    colors_path: str,
    max_points: int,
    point_size: float,
    background: str,
) -> tuple[str, str]:
    selected_points_path = Path(points_path).expanduser()
    if not selected_points_path.exists() and selected_points_path == DEFAULT_POINTS_PATH and DEFAULT_OBJ_PATH.exists():
        selected_points_path = DEFAULT_OBJ_PATH

    try:
        points, colors = load_point_cloud(str(selected_points_path), colors_path)
    except Exception as exc:
        message = (
            "No point cloud is ready yet. "
            "Run `conda run -n DIP python bundle_adjustment.py` first, "
            "or enter the path to an existing `.npy` / `.obj` point-cloud file."
        )
        return build_message_card(message), f"Viewer not loaded: {exc}"

    total_points = len(points)
    points, colors = sample_point_cloud(points, colors, int(max_points))
    points = normalize_points(points)

    srcdoc = build_viewer_srcdoc(
        points=points,
        colors=colors,
        point_size=point_size,
        background=background,
    )
    viewer = build_iframe_html(srcdoc)
    status = (
        f"Loaded {len(points):,} / {total_points:,} points from {selected_points_path}. "
        "Drag inside the viewer to orbit and use the mouse wheel to zoom."
    )
    return viewer, status


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Free Camera Point-Cloud Viewer") as demo:
        gr.Markdown(
            """
            # Free Camera Point-Cloud Viewer
            Load the reconstructed point cloud and explore it interactively with orbit controls.
            The default path expects output generated by `bundle_adjustment.py`.
            """
        )

        with gr.Row():
            points_path = gr.Textbox(
                label="Point Cloud Path",
                value=str(DEFAULT_POINTS_PATH),
                info="Use a `.npy` Nx3 point array or an OBJ with vertex lines.",
            )
            colors_path = gr.Textbox(
                label="Color Path",
                value=str(DEFAULT_COLORS_PATH),
                info="Used when the point cloud file does not already contain colors.",
            )

        with gr.Row():
            max_points = gr.Slider(
                label="Max Displayed Points",
                minimum=500,
                maximum=20000,
                value=10000,
                step=500,
            )
            point_size = gr.Slider(
                label="Base Point Size",
                minimum=0.1,
                maximum=6.0,
                value=0.4,
                step=0.1,
            )
            background = gr.Textbox(
                label="Background Color",
                value="#0f1720",
                info="Any CSS color string works, for example `#101820` or `black`.",
            )

        load_button = gr.Button("Load Viewer", variant="primary")
        status = gr.Markdown()
        viewer = gr.HTML()

        load_button.click(
            fn=build_viewer,
            inputs=[points_path, colors_path, max_points, point_size, background],
            outputs=[viewer, status],
        )

        demo.load(
            fn=build_viewer,
            inputs=[points_path, colors_path, max_points, point_size, background],
            outputs=[viewer, status],
        )

        gr.Markdown(
            f"""
            Default files:
            - Point cloud: `{DEFAULT_POINTS_PATH}`
            - OBJ fallback: `{DEFAULT_OBJ_PATH}`
            - Colors: `{DEFAULT_COLORS_PATH}`
            """
        )

    return demo


def main() -> None:
    args = parse_args()
    demo = build_demo()
    server_port = args.server_port if args.server_port is not None else find_free_port()
    demo.launch(server_name=args.server_name, server_port=server_port, share=args.share)


if __name__ == "__main__":
    main()
