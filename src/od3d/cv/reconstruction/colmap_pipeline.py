import os
import shutil
from pathlib import Path

import numpy as np
import pycolmap
import torch
from od3d.io import run_cmd
from PIL import Image


def create_intrinsic_matrix(params):
    """
    Create the intrinsic camera matrix from given parameters.

    Parameters:
    params (list or np.ndarray): Camera parameters [focal length, cx, cy, radial distortion k]

    Returns:
    np.ndarray: The 3x3 intrinsic matrix.
    """
    f, cx, cy, k = params
    intrinsic_matrix = np.array(
        [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ]
    )
    return intrinsic_matrix


class ColmapPipeline:
    def __init__(
        self,
        image_dir,
        output_dir,
        file_num,
        ratio,
        start_frame_id,
        flip_sfm,
    ):
        import shutil
        colmap_exe = shutil.which("colmap")
        if colmap_exe is None:
            #colmap_exe="/home/stud/jiso/miniconda3/envs/r2d2/bin/colmap"
            raise RuntimeError("COLMAP not found in PATH. Activate environment or install it.")
        
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.colmap_exe = colmap_exe

        self.start_frame_id = start_frame_id
        self.flip_sfm = flip_sfm

        self.sfm_dir = self.output_dir.joinpath(
            f"Partial_Ratio_{100* ratio}_Percent_start_frame_{self.start_frame_id}_flip_sfm_{self.flip_sfm}"
        )

        self.sfm_dir_recon_sparse = self.sfm_dir.joinpath(f"sparse")
        self.sfm_dir_img = self.sfm_dir.joinpath(f"images")
        self.sfm_dir_recon_dense = self.sfm_dir.joinpath(f"dense")

        self.sfm_dir.mkdir(parents=True, exist_ok=True)
        self.sfm_dir_recon_sparse.mkdir(parents=True, exist_ok=True)
        self.sfm_dir_img.mkdir(parents=True, exist_ok=True)

        frame_list = []
        if not self.flip_sfm:
            for i in range(self.start_frame_id + 1, self.start_frame_id + file_num + 1):
                filename = f"frame{str(i).zfill(6)}.jpg"
                source_path = image_dir.joinpath("images").joinpath(filename)
                destination_path = self.sfm_dir_img.joinpath(filename)
                try:
                    shutil.copy(source_path, destination_path)
                    frame_list.append(Image.open(source_path))
                    print(
                        f"File copied successfully from {source_path} to {destination_path}"
                    )
                except OSError as e:
                    print(f"Unable to copy file. {e}")
                except Exception as e:
                    print(f"Unexpected error: {e}")
            if len(frame_list) != 0:
                frame_list[0].save(
                    self.sfm_dir.joinpath("output.gif"),
                    save_all=True,
                    append_images=frame_list[1:],
                    optimize=False,
                    duration=200,
                    loop=0,
                )

        else:
            for i in range(self.start_frame_id + 1, self.start_frame_id + file_num + 1):
                filename = f"frame{str(i).zfill(6)}.jpg"
                source_path = image_dir.joinpath("images").joinpath(filename)
                destination_path = self.sfm_dir_img.joinpath(filename)
                try:
                    with Image.open(source_path) as img:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    frame_list.append(img)
                    img.save(destination_path)
                except OSError as e:
                    print(f"Unable to find input image. {e}")
                except Exception as e:
                    print(f"Unexpected error: {e}")
            frame_list[0].save(
                self.sfm_dir.joinpath("output.gif"),
                save_all=True,
                append_images=frame_list[1:],
                optimize=False,
                duration=200,
                loop=0,
            )

    def run_colmap(self, logger):
        # Correcting the path usage for the database file
        database_path = self.sfm_dir_recon_sparse.joinpath(
            "database.db"
        )  # Database file path
        mvs_path = self.sfm_dir_recon_sparse.joinpath("mvs")  # MVS directory

        # Ensure the output and MVS directories exist
        # os.makedirs(mvs_path, exist_ok=True)

        # Running the COLMAP processes
        # pycolmap.extract_features(database_path, self.sfm_dir_img)
        pycolmap.extract_features(
            database_path, self.sfm_dir_img, sift_options={"max_num_features": 10000}
        )
        pycolmap.match_exhaustive(database_path)
        maps = pycolmap.incremental_mapping(
            database_path, self.sfm_dir_img, self.sfm_dir_recon_sparse
        )
        if not maps:
            logger.error("COLMAP reconstruction failed. Check logs for details.")
            return
        reconstruction = pycolmap.Reconstruction(
            os.path.join(self.sfm_dir_recon_sparse, "0")
        )
        reconstruction.export_PLY(os.path.join(self.sfm_dir_recon_sparse, "sparse.ply"))

        self.export_camera_parameters(reconstruction)
        run_cmd(
            f"{self.colmap_exe} image_undistorter --image_path {self.sfm_dir_img} --input_path {self.sfm_dir_recon_sparse}/0 --output_path {self.sfm_dir_recon_dense} --output_type COLMAP --max_image_size 2000",
            logger=logger,
        )

        run_cmd(
            f"{self.colmap_exe} patch_match_stereo --workspace_path {self.sfm_dir_recon_dense} --workspace_format COLMAP --PatchMatchStereo.geom_consistency true",
            logger=logger,
        )

        run_cmd(
            f"{self.colmap_exe} stereo_fusion --workspace_path {self.sfm_dir_recon_dense} --workspace_format COLMAP --input_type geometric --output_path {self.sfm_dir}/dense.ply",
            logger=logger,
        )
        shutil.rmtree(
            self.sfm_dir_recon_dense.joinpath("stereo").joinpath("normal_maps")
        )
        self.save_ray_center_3d()

    def export_camera_parameters(self, reconstruction):
        self.sfm_dir_extrinsic = self.sfm_dir.joinpath("extrinsic")
        self.sfm_dir_intrinsic = self.sfm_dir.joinpath("intrinsic")

        os.makedirs(self.sfm_dir_extrinsic, exist_ok=True)
        os.makedirs(self.sfm_dir_intrinsic, exist_ok=True)

        for image_id, image in reconstruction.images.items():
            print(image_id, image)
            print(image.name)

            image_name = image.name
            camera_id = image.camera_id
            transformation = np.eye(4)
            rotation = image.cam_from_world.rotation.matrix()
            translation = image.cam_from_world.translation

            transformation[:3, :3] = rotation
            transformation[:3, -1] = translation
            # print('transformation matrix ', transformation)
            camera = reconstruction.cameras[camera_id]
            # print('camera ', camera)
            intrinsic = np.eye(4)
            intrinsic[:3, :3] = create_intrinsic_matrix(camera.params)
            # print('intrinsic ', intrinsic)
            print("----------------------------------")

            print("image name", image_name)
            print(
                "check",
                os.path.join(self.sfm_dir_extrinsic, image_name.split(".")[0] + ".pt"),
            )
            torch.save(
                torch.tensor(transformation),
                os.path.join(self.sfm_dir_extrinsic, image_name.split(".")[0] + ".pt"),
            )
            torch.save(
                torch.tensor(intrinsic),
                os.path.join(self.sfm_dir_intrinsic, image_name.split(".")[0] + ".pt"),
            )

    def save_ray_center_3d(self):
        extrinsic_list = []
        intrinsic_list = []
        for ex_name, in_name in zip(
            sorted(os.listdir(self.sfm_dir_extrinsic)),
            sorted(os.listdir(self.sfm_dir_intrinsic)),
        ):
            print("ex name ", ex_name)
            print("in name ", in_name)
            print("---------------------------")
            extrinsic = torch.load(self.sfm_dir_extrinsic.joinpath(ex_name))
            intrinsic = torch.load(self.sfm_dir_intrinsic.joinpath(in_name))
            extrinsic_list.append(extrinsic)
            intrinsic_list.append(intrinsic)

        extrinsic_list_tensor = torch.stack(extrinsic_list, dim=0)
        intrinsic_list_tensor = torch.stack(intrinsic_list, dim=0)

        from od3d.cv.geometry.fit.rays_center3d import fit_rays_center3d

        center3d = fit_rays_center3d(cams_tform4x4_obj=extrinsic_list_tensor)

        torch.save(center3d.detach().cpu(), f=self.sfm_dir.joinpath("rays_center3d.pt"))
