import os
import torch
import torch as th
import numpy as np
import cv2
import logging
import argparse
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange, repeat
from torchvision import transforms
from rose import RoSEUNetSpatioTemporalConditionModel, RoSEDiffusionPipeline
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_gray(org_path):
    org = cv2.imread(str(org_path))
    gray = np.mean(org, axis=2, keepdims=True).astype(np.uint8)
    org = np.repeat(gray, 3, axis=2)
    return org


@th.no_grad()
def inference(
    pipeline,
    img_path,
    input_root,
    output_root,
    frame_num,
    target_size=(576, 576),
    seed=42,
    **kwargs
):
    device = pipeline.device
    
    # 1. Prepare Image
    img_np = prepare_gray(img_path)
    validation_img = Image.fromarray(img_np.astype(np.uint8)).resize(target_size, Image.NEAREST)

    # 2. Setup Lighting & Angles
    i = th.arange(frame_num, dtype=th.float32)
    azimuth = ((i / frame_num) * 360 % 360).deg2rad()
    elevation = th.tensor([45] * frame_num).deg2rad()

    lgt_dir = th.stack([
        th.cos(elevation) * th.sin(azimuth),
        th.cos(elevation) * th.cos(azimuth),
        th.sin(elevation)
    ], dim=1)
    lgt_dir = rearrange(lgt_dir, 'f c -> f c 1 1')

    elevations_deg = [0] * frame_num
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    azimuths_rad = elevations_deg.copy()

    # 3. Pipeline Inference
    generator = th.Generator(device=device)
    generator.manual_seed(seed)

    video_latents = pipeline(
        image=validation_img,
        width=target_size[0],
        height=target_size[1],
        num_frames=frame_num,
        num_inference_steps=25,
        decode_chunk_size=8,
        polars_rad=polars_rad,
        azimuths_rad=azimuths_rad,
        output_type="pil",
        generator=generator
    ).frames[0]

    # 4. Post-process to extract Shading and Normals
    to_tensor = transforms.ToTensor()
    tensor_list = [to_tensor(img) for img in video_latents]
    video_latents = th.stack(tensor_list, dim=0).to(device)

    shading = video_latents.mean(dim=1)
    _, H, W = shading.shape
    shading = rearrange(shading, 'b h w -> (h w) b 1')

    lgt_dir_s = repeat(lgt_dir, 'f c 1 1 -> (h w) f c', h=H, w=W).to(device)

    shading = shading.clip(0)
    shading = shading / shading.max()

    mask = (shading > 0).to(shading.dtype)

    # Photometric Stereo Least Squares Solve
    M = th.matmul(lgt_dir_s.transpose(1, 2), mask * lgt_dir_s)
    y = th.matmul(lgt_dir_s.transpose(1, 2), mask * shading)
    epsilon = 1e-6
    M = M + epsilon * th.eye(M.shape[-1], device=M.device).unsqueeze(0)
    x = th.linalg.solve(M, y)

    shading_nml = th.nn.functional.normalize(x, dim=1, p=2)
    shading_nml = rearrange(shading_nml, '(h w) c 1 -> h w c', h=H, w=W).detach().cpu().numpy()
    shading_nml = (shading_nml * 127.5 + 128).astype(np.uint8)
    shading_nml_img = Image.fromarray(shading_nml)

    # 5. Save Results
    rel_path = img_path.relative_to(input_root)
    out_file = rel_path.with_name(rel_path.stem + ".png")
    output_path = Path(output_root) / out_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    shading_nml_img.save(str(output_path))
    logger.info(f"Saved normal map -> {output_path}")


def load_primary_models(pretrained_model_path, ckpt_dir):
    logger.info(f"Loading primary models...")
    logger.info(f"Base path: {pretrained_model_path}")
    logger.info(f"UNet checkpoint: {ckpt_dir}")
    
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = RoSEUNetSpatioTemporalConditionModel.from_pretrained(ckpt_dir, subfolder="unet")
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_path, subfolder="feature_extractor")
    
    pipeline = RoSEDiffusionPipeline(
        image_encoder=image_encoder, 
        feature_extractor=feature_extractor, 
        unet=unet, 
        vae=vae,
        scheduler=noise_scheduler,
    )
    return pipeline, unet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_num", type=int, default=9)
    parser.add_argument("--input_dir", type=str, default=None, help="Path to the input directory")
    parser.add_argument("--output_dir", type=str, default="./results", help="Path to save results")
    parser.add_argument("--src_path", type=str, default=None, help="Path to the UNet checkpoints directory")
    parser.add_argument("--pretrained_model_path", type=str, default="chenguolin/sv3d-diffusers", help="Path to base pretrained models")
    parser.add_argument("--target_size", type=int, nargs=2, default=[576, 576])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Model Loading & Device Allocation
    pipeline, unet = load_primary_models(args.pretrained_model_path, args.src_path)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    weight_dtype = th.float16
    pipeline = pipeline.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    logger.info(f"GPU memory allocated before inference: {th.cuda.memory_allocated()/1024**2:.2f} MB")
    logger.info(f"GPU memory reserved before inference: {th.cuda.memory_reserved()/1024**2:.2f} MB")

    # Locate Images
    input_root = Path(args.input_dir)
    allowed_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_files = sorted([p for p in input_root.rglob("*") if p.suffix.lower() in allowed_ext])
    
    if not image_files:
        logger.error(f"No images found in {args.input_dir}")
        exit(1)
        
    logger.info(f"Found {len(image_files)} images for processing.")

    # Main Processing Loop
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            inference(
                pipeline=pipeline,
                img_path=img_path,
                input_root=input_root,
                output_root=args.output_dir,
                frame_num=args.frame_num,
                target_size=tuple(args.target_size),
                seed=args.seed
            )
            logger.info(f"GPU memory allocated after {img_path.name}: {th.cuda.memory_allocated()/1024**2:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")

    logger.info("All Done.")