#!/usr/bin/env python3
"""
COLMAP 3D reconstruction pipeline (Python version)
Usage: python run_colmap.py
"""

import subprocess
import sys
from pathlib import Path

# ============================================================
# 请在此处设置 colmap 可执行文件的完整路径
# 例如 Linux/macOS:  "/usr/local/bin/colmap"
#       Windows:      "C:/Program Files/COLMAP/bin/colmap.bat"
# ============================================================
COLMAP_EXE = Path("D:/project/DIP-Workspace/03_Bundle Adjustment/colmap-x64-windows-cuda/COLMAP.bat")

def run_command(args: list[str], description: str = "") -> None:
    """运行外部命令，失败时退出程序。"""
    if description:
        print(description)
    # 第一个参数使用自定义的 colmap 路径
    cmd = [str(COLMAP_EXE)] + args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

def main():
    # 路径设置
    DATASET_PATH = Path("data")
    IMAGE_PATH = DATASET_PATH / "images"
    COLMAP_PATH = DATASET_PATH / "colmap"

    # 创建必要的目录
    (COLMAP_PATH / "sparse").mkdir(parents=True, exist_ok=True)
    (COLMAP_PATH / "dense").mkdir(parents=True, exist_ok=True)

    # 第1步：特征提取
    run_command([
        "feature_extractor",
        "--database_path", str(COLMAP_PATH / "database.db"),
        "--image_path", str(IMAGE_PATH),
        "--ImageReader.camera_model", "PINHOLE",
        "--ImageReader.single_camera", "1"
    ], "=== Step 1: Feature Extraction ===")

    # 第2步：特征匹配（穷举匹配）
    run_command([
        "exhaustive_matcher",
        "--database_path", str(COLMAP_PATH / "database.db")
    ], "=== Step 2: Feature Matching ===")

    # 第3步：稀疏重建（捆绑调整）
    run_command([
        "mapper",
        "--database_path", str(COLMAP_PATH / "database.db"),
        "--image_path", str(IMAGE_PATH),
        "--output_path", str(COLMAP_PATH / "sparse")
    ], "=== Step 3: Sparse Reconstruction (Bundle Adjustment) ===")

    # 第4步：图像去畸变
    run_command([
        "image_undistorter",
        "--image_path", str(IMAGE_PATH),
        "--input_path", str(COLMAP_PATH / "sparse" / "0"),
        "--output_path", str(COLMAP_PATH / "dense")
    ], "=== Step 4: Image Undistortion ===")

    # 第5步：稠密重建（Patch Match 立体匹配）
    run_command([
        "patch_match_stereo",
        "--workspace_path", str(COLMAP_PATH / "dense")
    ], "=== Step 5: Dense Reconstruction (Patch Match Stereo) ===")

    # 第6步：立体融合，生成点云
    run_command([
        "stereo_fusion",
        "--workspace_path", str(COLMAP_PATH / "dense"),
        "--output_path", str(COLMAP_PATH / "dense" / "fused.ply")
    ], "=== Step 6: Stereo Fusion ===")

    # 完成
    print("\n=== Done! ===")
    print("Results:")
    print(f"  Sparse: {COLMAP_PATH}/sparse/0/")
    print(f"  Dense:  {COLMAP_PATH}/dense/fused.ply")

if __name__ == "__main__":
    main()