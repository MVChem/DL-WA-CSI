import os
import random
from datetime import datetime
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from mrspy.plot import plot_chemicalshift_image, SpecPlotter

from model.fca_unet import UNet
from util.util import log_loss, random_noise_level, simulation
from util.dataset import MRSDataset


def main(args):
    now = datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M")
    log_dir = f"{args.log_dir}/{args.project_name}/{now}"
    os.makedirs(log_dir, exist_ok=True)

    model = UNet(
        sample_size=(32, 32),
        block_out_channels=(72, 72, 72, 72),
        in_channels=72,
        out_channels=72,
        cross_attention_dim=256,
    )

    train_dataset = MRSDataset(txt_file=args.train_txt_path)
    test_dataset = MRSDataset(txt_file=args.test_txt_path)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    device = torch.device(args.device)
    model.to(device)

    l2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Train)")
        for batch_idx, inputs in enumerate(progress_bar):
            img1 = inputs[0]
            img2 = inputs[1]
            img3 = inputs[2]

            noise_level = random_noise_level(args.noise_level)
            res = simulation(img1, img2, img3, device, noise_level)
            gt = res["gt"]
            wei_no = res["wei_no"]

            b, t, l, w, h = wei_no.shape
            reference_imgs = torch.zeros([b, 256, 256], device=device)

            outputs = model(wei_no, reference_imgs)
            loss = l2(outputs, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b
            total += b
            progress_bar.set_postfix(loss=train_loss / total)

        avg_train_loss = train_loss / total if total > 0 else 0.0

        model.eval()
        test_loss = 0.0
        total_test = 0

        with torch.no_grad():
            num_test_batches = len(test_loader)
            progress_bar_test = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Test)")
            plot_idx = random.randint(0, num_test_batches - 1)

            for batch_idx, inputs in enumerate(progress_bar_test):
                img1 = inputs[0]
                img2 = inputs[1]
                img3 = inputs[2]

                noise_level = random_noise_level(args.noise_level)
                res = simulation(img1, img2, img3, device, noise_level)
                gt = res["gt"]
                wei_no = res["wei_no"]

                b, t, l, w, h = wei_no.shape
                reference_imgs = torch.zeros([b, 256, 256], device=device)

                outputs = model(wei_no, reference_imgs)
                loss = l2(outputs, gt)

                if batch_idx == plot_idx:
                    epoch_dir = f"{log_dir}/{epoch}"
                    os.makedirs(epoch_dir, exist_ok=True)

                    shifts = SpecPlotter.from_tensor(
                        wei_no[0].cpu().reshape(t, l, w, h)
                    ).get_chemical_shifts(n=3)

                    plot_chemicalshift_image(
                        wei_no[0].cpu(),
                        cmap="hot",
                        chemicalshift=shifts[0],
                        save_path=f"{epoch_dir}/wei_no_n{noise_level:.3g}_p{plot_idx}_s{shifts[0][0]}_{shifts[0][1]}",
                        dpi=200,
                    )
                    plot_chemicalshift_image(
                        outputs[0].cpu(),
                        cmap="hot",
                        chemicalshift=shifts[0],
                        save_path=f"{epoch_dir}/pred_p{plot_idx}_s{shifts[0][0]}_{shifts[0][1]}",
                        dpi=200,
                    )
                    plot_chemicalshift_image(
                        gt[0].cpu(),
                        cmap="hot",
                        chemicalshift=shifts[0],
                        save_path=f"{epoch_dir}/gt_p{plot_idx}_s{shifts[0][0]}_{shifts[0][1]}",
                        dpi=200,
                    )

                    SpecPlotter.from_tensor(wei_no[0].cpu()).spec_plot(
                        plot_all=True,
                        save_path=f"{epoch_dir}/wei_no_{noise_level:.3g}_spec.jpg",
                    )
                    SpecPlotter.from_tensor(outputs[0].cpu()).spec_plot(
                        plot_all=True,
                        save_path=f"{epoch_dir}/pred_spec.jpg",
                    )
                    SpecPlotter.from_tensor(gt[0].cpu()).spec_plot(
                        plot_all=True,
                        save_path=f"{epoch_dir}/gt_spec.jpg",
                    )

                test_loss += loss.item() * b
                total_test += b
                progress_bar_test.set_postfix(loss=test_loss / total_test)

        avg_test_loss = test_loss / total_test if total_test > 0 else 0.0

        log_loss(epoch + 1, avg_train_loss, avg_test_loss, log_dir)
        torch.save(model.state_dict(), f"{log_dir}/unet_epoch{epoch+1}.pth")
        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {avg_train_loss:.6f}  "
            f"Test Loss: {avg_test_loss:.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt_path", type=str, default="data/hybrid_train.txt")
    parser.add_argument("--test_txt_path", type=str, default="data/hybrid_test.txt")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--project_name", type=str, default="WA")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--noise_level", type=float, default=1.6)
    args = parser.parse_args()
    main(args)
