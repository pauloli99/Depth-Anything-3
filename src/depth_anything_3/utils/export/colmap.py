# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pycolmap
import cv2 as cv
import numpy as np
import sqlite3

from PIL import Image

from depth_anything_3.specs import Prediction
from depth_anything_3.utils.logger import logger

from .glb import _depths_to_world_points_with_colors


def export_to_colmap(
    prediction: Prediction,
    export_dir: str,
    image_paths: list[str],
    conf_thresh_percentile: float = 40.0,
    process_res_method: str = "upper_bound_resize",
) -> None:
    # 1. Data preparation
    conf_thresh = np.percentile(prediction.conf, conf_thresh_percentile)
    points, colors = _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        prediction.processed_images,
        prediction.conf,
        conf_thresh,
    )
    num_points = len(points)
    logger.info(f"Exporting to COLMAP with {num_points} points")
    num_frames = len(prediction.processed_images)
    h, w = prediction.processed_images.shape[1:3]
    points_xyf = _create_xyf(num_frames, h, w)
    points_xyf = points_xyf[prediction.conf >= conf_thresh]

    # 2. Set Reconstruction
    reconstruction = pycolmap.Reconstruction()

    point3d_ids = []
    for vidx in range(num_points):
        point3d_id = reconstruction.add_point3D(points[vidx], pycolmap.Track(), colors[vidx])
        point3d_ids.append(point3d_id)

    for fidx in range(num_frames):
        orig_w, orig_h = Image.open(image_paths[fidx]).size

        intrinsic = prediction.intrinsics[fidx]
        if process_res_method.endswith("resize"):
            intrinsic[:1] *= orig_w / w
            intrinsic[1:2] *= orig_h / h
        elif process_res_method == "crop":
            raise NotImplementedError("COLMAP export for crop method is not implemented")
        else:
            raise ValueError(f"Unknown process_res_method: {process_res_method}")

        pycolmap_intri = np.array(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
        )

        extrinsic = prediction.extrinsics[fidx]
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(extrinsic[:3, :3]), extrinsic[:3, 3])

        # set and add camera
        camera = pycolmap.Camera()
        camera.camera_id = fidx + 1
        camera.model = pycolmap.CameraModelId.PINHOLE
        camera.width = orig_w
        camera.height = orig_h
        camera.params = pycolmap_intri
        reconstruction.add_camera(camera)

        # set and add rig (from camera)
        rig = pycolmap.Rig()
        rig.rig_id = camera.camera_id
        rig.add_ref_sensor(camera.sensor_id)
        reconstruction.add_rig(rig)

        # set image
        image = pycolmap.Image()
        image.image_id = fidx + 1
        image.camera_id = camera.camera_id

        # set and add frame (from image)
        frame = pycolmap.Frame()
        frame.frame_id = image.image_id
        frame.rig_id = camera.camera_id
        frame.add_data_id(image.data_id)
        frame.rig_from_world = cam_from_world
        reconstruction.add_frame(frame)

        # set point2d and update track
        point2d_list = []
        points_in_frame = points_xyf[:, 2].astype(np.int32) == fidx
        for vidx in np.where(points_in_frame)[0]:
            point2d = points_xyf[vidx][:2]
            point2d[0] *= orig_w / w
            point2d[1] *= orig_h / h
            point3d_id = point3d_ids[vidx]
            point2d_list.append(pycolmap.Point2D(point2d, point3d_id))
            reconstruction.point3D(point3d_id).track.add_element(
                image.image_id, len(point2d_list) - 1
            )

        # set and add image
        image.frame_id = image.image_id
        image.name = os.path.basename(image_paths[fidx])
        image.points2D = pycolmap.Point2DList(point2d_list)
        reconstruction.add_image(image)

    # 3. Export
    # Create sparse/0/ directory structure (COLMAP standard)
    sparse_dir = os.path.join(export_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    # Write reconstruction to sparse/0/
    reconstruction.write(sparse_dir)

    # Create project.ini and database.db files in sparse/0/
    _create_project_files(sparse_dir, export_dir, image_paths)


def _create_xyf(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices (fidx) for all frames.
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.int32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.int32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf


def _create_project_files(sparse_dir: str, export_dir: str, image_paths: list[str]) -> None:
    """
    Create project.ini file and empty database.db for COLMAP GUI compatibility.

    Args:
        sparse_dir: Sparse reconstruction directory (sparse/0/)
        export_dir: Export directory (parent of sparse/)
        image_paths: List of image file paths
    """
    # Find the images directory
    images_dir = None
    for path in image_paths:
        # Check if path contains 'images' directory
        if "images" in path:
            images_dir = os.path.dirname(path)
            break

    if images_dir is None:
        # Fallback: look for images directory in export_dir
        images_dir = os.path.join(export_dir, "images")

    # Create empty database.db file in sparse/0/ (COLMAP SQLite database)
    db_path = os.path.join(sparse_dir, "database.db")
    _create_empty_colmap_database(db_path, image_paths)

    # Create project.ini content (relative paths from sparse/0/)
    project_ini_content = f"""[General]
database_path=database.db
image_path={os.path.relpath(images_dir, sparse_dir)}

[Reconstruction]
reconstruction_path=.
"""

    # Write project.ini in sparse/0/
    project_ini_path = os.path.join(sparse_dir, "project.ini")
    with open(project_ini_path, "w") as f:
        f.write(project_ini_content)

    logger.info(f"Created project.ini file: {project_ini_path}")
    logger.info(f"Created empty database.db file: {db_path}")


def _create_empty_colmap_database(db_path: str, image_paths: list[str]) -> None:
    """
    Create an empty COLMAP database with basic tables and image entries.

    Args:
        db_path: Path to create the database
        image_paths: List of image file paths
    """
    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create COLMAP database tables (simplified version)
    cursor.execute(
        """
        CREATE TABLE cameras (
            camera_id INTEGER PRIMARY KEY,
            model INTEGER,
            width INTEGER,
            height INTEGER,
            params BLOB,
            prior_focal_length INTEGER
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE images (
            image_id INTEGER PRIMARY KEY,
            name TEXT,
            camera_id INTEGER,
            prior_qw REAL,
            prior_qx REAL,
            prior_qy REAL,
            prior_qz REAL,
            prior_tx REAL,
            prior_ty REAL,
            prior_tz REAL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE keypoints (
            image_id INTEGER,
            rows INTEGER,
            cols INTEGER,
            data BLOB
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE descriptors (
            image_id INTEGER,
            rows INTEGER,
            cols INTEGER,
            data BLOB
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE matches (
            pair_id INTEGER,
            rows INTEGER,
            cols INTEGER,
            data BLOB
        )
    """
    )

    # Add image entries (without feature data, just names)
    for i, image_path in enumerate(image_paths, 1):
        image_name = os.path.basename(image_path)
        cursor.execute(
            "INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)",
            (i, image_name, 1),  # camera_id will be updated when cameras are imported
        )

    conn.commit()
    conn.close()
