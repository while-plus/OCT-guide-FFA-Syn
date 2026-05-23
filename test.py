import argparse
import os

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.vitunet import ViTUNetGenerator
from trainer.datasets import CSVDataset


MODEL_ARGS = {
    "image_shape": (3, 512, 512),
    "features": 384,
    "n_heads": 6,
    "n_blocks": 6,
    "ffn_features": 1536,
    "embed_features": 384,
    "activ": "gelu",
    "norm": "layer",
    "unet_features_list": [48, 96, 192, 384],
    "unet_activ": "leakyrelu",
    "unet_norm": "instance",
    "unet_downsample": "conv",
    "unet_upsample": "upsample-conv",
    "rezero": True,
    "activ_output": "sigmoid",
    "out_ch": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for Reg-GAN test samples.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--save-image-dir", default="./outputs/test", help="Directory to save generated images.")
    parser.add_argument("--ckpt-path", default="./netG_A2B.pth", help="Path to generator checkpoint.")
    parser.add_argument("--csv-path", default="./data/samples.csv", help="CSV file describing samples.")
    parser.add_argument("--data-dir", default="./data", help="Root directory for image files referenced by the CSV.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader worker count.")
    return parser.parse_args()


def tensor_to_array(image_tensor):
    array = image_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    if array.shape[-1] == 1:
        array = array[..., 0]
    return array


def build_dataloader(csv_path, data_dir, batch_size, num_workers):
    dataset = CSVDataset(csv_path, data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_generator(ckpt_path, device):
    model = ViTUNetGenerator(**MODEL_ARGS)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k[7:]: v for k, v in ckpt.items()} if all(k.startswith("module.") for k in ckpt.keys()) else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def save_predictions(batch_image_files, fake_images, save_image_dir):
    saved_count = 0
    for image_file, gen_image in zip(batch_image_files, tensor_to_array(fake_images)):
        output_rel_path = os.path.splitext(image_file)[0] + ".jpg"
        output_path = os.path.join(save_image_dir, output_rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if gen_image.ndim == 3:
            gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, gen_image)
        saved_count += 1
    return saved_count


def run_inference(args):
    dataloader = build_dataloader(args.csv_path, args.data_dir, args.batch_size, args.num_workers)
    model = load_generator(args.ckpt_path, args.device)
    total_saved = 0

    print("Inference Settings")
    print(f"  Device         : {args.device}")
    print(f"  Checkpoint     : {args.ckpt_path}")
    print(f"  CSV Path       : {args.csv_path}")
    print(f"  Data Dir       : {args.data_dir}")
    print(f"  Save Image Dir : {args.save_image_dir}")
    print(f"  Batch Size     : {args.batch_size}")
    print(f"  Num Workers    : {args.num_workers}")

    for batch in tqdm(dataloader, total=len(dataloader)):
        real_A = batch["A"].to(args.device)
        oct_images = batch["OCT"].to(args.device)
        line_cords = batch["Line Cords"].to(args.device)

        with torch.no_grad():
            fake_B, _ = model(real_A, oct_images, line_cords)

        total_saved += save_predictions(batch["image file"], fake_B, args.save_image_dir)

    print("Inference Completed")
    print(f"  Saved Images   : {total_saved}")
    print(f"  Output Dir     : {args.save_image_dir}")


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
