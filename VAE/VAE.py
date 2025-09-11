"""
Minimal but solid VAE in PyTorch (MNIST or CIFAR-10)
- CNN encoder/decoder
- ELBO with BCE reconstruction + KL (β-VAE supported)
- Deterministic eval, image sampling & checkpointing

Usage (examples):
  python VAE_PyTorch_minimal_but_solid.py --dataset mnist --epochs 10
  python VAE_PyTorch_minimal_but_solid.py --dataset cifar10 --epochs 50 --latent-dim 128 --beta 1.0

Notes for fast sanity check:
  - MNIST: ~1-2 mins/epoch on GPU; reconstruction should look reasonable after ~5 epochs.
  - CIFAR-10: needs more capacity/epochs for good samples; this is an educational baseline, not SOTA.
"""

import os
import math
import argparse
from dataclasses import dataclass
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils



def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def num_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class Encoder(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int, base_ch: int = 64, img_size: int = 28):
        super().__init__()
        s = img_size
        # Downsampling to 4x4 feature map (works for 28 or 32 with appropriate padding/stride)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),  # s -> s/2
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1),  # s/2 -> s/4
            nn.BatchNorm2d(base_ch*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1),  # s/4 -> s/8
            nn.BatchNorm2d(base_ch*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # compute final spatial size after 3 downsamples
        self.spatial = s // 8
        hid = base_ch * 4 * self.spatial * self.spatial
        self.fc_mu = nn.Linear(hid, latent_dim)
        self.fc_logvar = nn.Linear(hid, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_ch: int, latent_dim: int, base_ch: int = 64, img_size: int = 28):
        super().__init__()
        s = img_size // 8
        hid = base_ch * 4 * s * s
        self.fc = nn.Linear(latent_dim, hid)
        self.s = s
        self.base_ch = base_ch
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1),  # s -> 2s
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1),  # 2s -> 4s
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch, out_ch, 4, 2, 1),  # 4s -> 8s (= img_size)
            # output are logits; we apply sigmoid in loss via BCEWithLogits
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), self.base_ch*4, self.s, self.s)
        x_logits = self.net(h)
        return x_logits


class VAE(nn.Module):
    def __init__(self, in_ch: int, latent_dim: int, base_ch: int = 64, img_size: int = 28):
        super().__init__()
        self.encoder = Encoder(in_ch, latent_dim, base_ch, img_size)
        self.decoder = Decoder(in_ch, latent_dim, base_ch, img_size)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decode(z)
        return x_logits, mu, logvar


def elbo_loss(x_logits, x, mu, logvar, beta=1.0):
    # BCE with logits expects inputs in [0,1] target; we assume input x already normalized to [0,1]
    bce = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')
    # KL divergence between q(z|x)=N(mu, sigma) and p(z)=N(0, I)
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = bce + beta * kl
    return loss, bce, kl


def get_dataloaders(dataset: str, data_dir: str, batch_size: int, img_size: int):
    if dataset == 'mnist':
        in_ch = 1
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=tfm)
        test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)
    elif dataset == 'cifar10':
        in_ch = 3
        tfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm)
        test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm)
    else:
        raise ValueError('Unsupported dataset: choose mnist or cifar10')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader, in_ch


def save_samples(decoder, epoch, out_dir, device, nrow=8, n=64):
    decoder.eval()
    with torch.no_grad():
        z = torch.randn(n, decoder.fc.in_features, device=device)
        x_logits = decoder(z)
        x = torch.sigmoid(x_logits)
        grid = vutils.make_grid(x, nrow=nrow, padding=2)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"samples_ep{epoch:03d}.png")
        vutils.save_image(grid, path)
        return path


def save_recon(model, batch, epoch, out_dir, device, n=8):
    model.eval()
    x = batch[0:n].to(device)
    with torch.no_grad():
        x_logits, _, _ = model(x)
        x_rec = torch.sigmoid(x_logits)
        comp = torch.cat([x, x_rec], dim=0)
        grid = vutils.make_grid(comp, nrow=n, padding=2)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"recon_ep{epoch:03d}.png")
        vutils.save_image(grid, path)
        return path


def train(cfg):
    set_seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not cfg.cpu else 'cpu')

    train_loader, test_loader, in_ch = get_dataloaders(cfg.dataset, cfg.data_dir, cfg.batch_size, cfg.img_size)

    model = VAE(in_ch=in_ch, latent_dim=cfg.latent_dim, base_ch=cfg.base_ch, img_size=cfg.img_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(f"Model params: {num_params(model)/1e6:.2f}M | device={device}")

    global_step = 0
    best_val = math.inf

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_loss = run_bce = run_kl = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            x_logits, mu, logvar = model(x)
            loss, bce, kl = elbo_loss(x_logits, x, mu, logvar, beta=cfg.beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            run_loss += loss.item()
            run_bce += bce.item()
            run_kl += kl.item()
            global_step += 1

        n = len(train_loader.dataset)
        print(f"Epoch {epoch:03d} | train loss/elbo: {run_loss/n:.4f} | recon: {run_bce/n:.4f} | kl: {run_kl/n:.4f}")

        # Validation (recon only as quick proxy)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_logits, mu, logvar = model(x)
                loss, _, _ = elbo_loss(x_logits, x, mu, logvar, beta=cfg.beta)
                val_loss += loss.item()
        val_loss /= len(test_loader.dataset)
        print(f"           |   val elbo: {val_loss:.4f}")

        os.makedirs(cfg.out_dir, exist_ok=True)
        # Save reconstructions and samples every epoch
        # Grab a small batch from train loader for visualization
        x_vis, _ = next(iter(train_loader))
        recon_path = save_recon(model, x_vis, epoch, cfg.out_dir, device)
        sample_path = save_samples(model.decoder, epoch, cfg.out_dir, device)
        print(f"Saved recon -> {recon_path} | samples -> {sample_path}")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(cfg.out_dir, 'vae_best.pt')
            torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, ckpt_path)
            print(f"[best] checkpoint saved -> {ckpt_path}")

    # final checkpoint
    last_path = os.path.join(cfg.out_dir, 'vae_last.pt')
    torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, last_path)
    print(f"[last] checkpoint saved -> {last_path}")



@dataclass
class Config:
    dataset: str = 'mnist'  # 'mnist' or 'cifar10'
    data_dir: str = './data'
    out_dir: str = './runs/vae'
    epochs: int = 10
    batch_size: int = 128
    lr: float = 2e-3
    latent_dim: int = 64
    base_ch: int = 64
    img_size: int = 28  # 28 for MNIST, 32 for CIFAR-10
    beta: float = 1.0   # beta-VAE coefficient
    grad_clip: float = 1.0
    seed: int = 42
    cpu: bool = False


def parse_args():
    p = argparse.ArgumentParser(description='Minimal but solid VAE (PyTorch)')
    p.add_argument('--dataset', type=str, default=Config.dataset, choices=['mnist', 'cifar10'])
    p.add_argument('--data-dir', type=str, default=Config.data_dir)
    p.add_argument('--out-dir', type=str, default=Config.out_dir)
    p.add_argument('--epochs', type=int, default=Config.epochs)
    p.add_argument('--batch-size', type=int, default=Config.batch_size)
    p.add_argument('--lr', type=float, default=Config.lr)
    p.add_argument('--latent-dim', type=int, default=Config.latent_dim)
    p.add_argument('--base-ch', type=int, default=Config.base_ch)
    p.add_argument('--img-size', type=int, default=Config.img_size)
    p.add_argument('--beta', type=float, default=Config.beta)
    p.add_argument('--grad-clip', type=float, default=Config.grad_clip)
    p.add_argument('--seed', type=int, default=Config.seed)
    p.add_argument('--cpu', action='store_true', help='force CPU')
    args = p.parse_args()
    cfg = Config(**vars(args))

    # auto-set img_size if dataset=cifar10 and user kept default 28
    if cfg.dataset == 'cifar10' and args.img_size == 28:
        cfg.img_size = 32
    return cfg


if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)
