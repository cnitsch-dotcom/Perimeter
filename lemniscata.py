#!/usr/bin/env python3

import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

CONFIG_PATH = Path(__file__).with_name('lemniscata_config.json')
LEMNISCATE_PERIMETER_FACTOR = 5.24411510858424


def load_config():
    with CONFIG_PATH.open('r', encoding='utf-8') as config_file:
        return json.load(config_file)


config = load_config()

seed = config.get('seed')
if seed is None:
    seed = int(time.time())
print(f"Random Seed: {seed}")
torch.manual_seed(seed)
np.random.seed(seed)


class VectorFieldNetwork(nn.Module):
    def __init__(self, hidden_dim=128, layers=4):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)
        )
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.activation = nn.Tanh()

    def forward(self, x):
        out = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        v = self.output_layer(out)

        # Keep the vector field inside the unit ball with a smooth projection.
        v_norm = torch.norm(v, dim=1, keepdim=True)
        scale = torch.tanh(v_norm) / (v_norm + 1e-6)
        return v * scale


def get_divergence(phi, x):
    phi_x = phi[:, 0]
    phi_y = phi[:, 1]

    grad_x = torch.autograd.grad(phi_x, x, torch.ones_like(phi_x), create_graph=True)[0][:, 0]
    grad_y = torch.autograd.grad(phi_y, x, torch.ones_like(phi_y), create_graph=True)[0][:, 1]
    return grad_x + grad_y


def is_inside_lemniscate(pts, scale):
    x = pts[:, 0]
    y = pts[:, 1]
    left_side = (x**2 + y**2) ** 2
    right_side = (scale**2) * (x**2 - y**2)
    return (x**2 >= y**2) & (left_side <= right_side)


def generate_boundary(scale, n_points=800):
    theta_right = np.linspace(-np.pi / 4, np.pi / 4, n_points)
    r_right = scale * np.sqrt(np.clip(np.cos(2 * theta_right), 0.0, None))
    x_right = r_right * np.cos(theta_right)
    y_right = r_right * np.sin(theta_right)

    theta_left = np.linspace(3 * np.pi / 4, 5 * np.pi / 4, n_points)
    r_left = scale * np.sqrt(np.clip(np.cos(2 * theta_left), 0.0, None))
    x_left = r_left * np.cos(theta_left)
    y_left = r_left * np.sin(theta_left)
    return x_right, y_right, x_left, y_left


def evaluate_perimeter(model, device, scale, eval_config):
    model.eval()

    x_min = eval_config['bounding_box']['x_min']
    x_max = eval_config['bounding_box']['x_max']
    y_min = eval_config['bounding_box']['y_min']
    y_max = eval_config['bounding_box']['y_max']
    n_samples = eval_config['n_samples']
    batch_size = eval_config['batch_size']
    area_box = (x_max - x_min) * (y_max - y_min)

    x_raw = torch.rand(n_samples * 2, 2, device=device)
    x_raw[:, 0] = x_raw[:, 0] * (x_max - x_min) + x_min
    x_raw[:, 1] = x_raw[:, 1] * (y_max - y_min) + y_min

    mask = is_inside_lemniscate(x_raw, scale)
    area_estimate = mask.float().mean().item() * area_box
    x_in = x_raw[mask][:n_samples]

    if len(x_in) < n_samples:
        print(f"Warning: requested {n_samples} samples but only found {len(x_in)} inside. Accuracy may suffer.")

    total_div = 0.0
    num_batches = 0

    for i in range(0, len(x_in), batch_size):
        batch = x_in[i:i + batch_size].clone().detach()
        batch.requires_grad_(True)

        phi = model(batch)
        div = get_divergence(phi, batch)

        total_div += torch.sum(div).item()
        num_batches += len(batch)

    avg_div = total_div / num_batches
    perimeter = avg_div * area_estimate
    return perimeter, area_estimate


def train():
    train_config = config['training']
    model_config = config['model']
    shape_config = config['shape']
    penalty_config = config['penalty']

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA acceleration")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    model = VectorFieldNetwork(
        hidden_dim=model_config['hidden_dim'],
        layers=model_config['layers'],
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    iterations = train_config['iterations']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    batch_size = train_config['batch_size']
    scale = shape_config['a']
    area_exact = scale ** 2
    perimeter_exact = LEMNISCATE_PERIMETER_FACTOR * scale
    penalty_multiplier = penalty_config['multiplier']

    print("Target: Bernoulli lemniscate")
    print(f"Scale parameter a: {scale:.5f}")
    print(f"Exact Area: {area_exact:.5f}")
    print(f"Exact Perimeter: {perimeter_exact:.5f}")
    print(f"Penalty Multiplier: {penalty_multiplier:.6f}")

    print("Starting training...")

    try:
        for i in range(iterations):
            model.train()

            x_min = train_config['bounding_box']['x_min']
            x_max = train_config['bounding_box']['x_max']
            y_min = train_config['bounding_box']['y_min']
            y_max = train_config['bounding_box']['y_max']

            x_raw = torch.rand(batch_size * train_config['oversample_factor'], 2, device=device)
            x_raw[:, 0] = x_raw[:, 0] * (x_max - x_min) + x_min
            x_raw[:, 1] = x_raw[:, 1] * (y_max - y_min) + y_min

            mask = is_inside_lemniscate(x_raw, scale)
            x_in = x_raw[mask][:batch_size]

            if len(x_in) < batch_size:
                continue

            x_in.requires_grad_(True)
            phi = model(x_in)
            div_phi = get_divergence(phi, x_in)

            perimeter_loss = -torch.mean(div_phi) * area_exact
            divergence_penalty = torch.mean(div_phi ** 2)
            loss = perimeter_loss + penalty_multiplier * divergence_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % train_config['log_every'] == 0:
                estimated_perimeter = torch.mean(div_phi).item() * area_exact
                print(
                    f"Iter {i}: Loss = {loss.item():.5f} | "
                    f"Est. P = {estimated_perimeter:.5f} | "
                    f"Penalty = {divergence_penalty.item():.5f}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted.")

    final_perimeter, estimated_area = evaluate_perimeter(model, device, scale, config['evaluation'])
    print("\nFinal Results:")
    print(f"Estimated Area: {estimated_area:.5f}")
    print(f"Analytical Area: {area_exact:.5f}")
    print(f"Area Absolute Error: {abs(estimated_area - area_exact):.5f}")
    print(f"Area Relative Error: {abs(estimated_area - area_exact) / area_exact * 100:.2f}%")
    print(f"Calculated Perimeter: {final_perimeter:.5f}")
    print(f"Analytical Perimeter: {perimeter_exact:.5f}")
    print(f"Absolute Error: {abs(final_perimeter - perimeter_exact):.5f}")
    print(f"Relative Error: {abs(final_perimeter - perimeter_exact) / perimeter_exact * 100:.2f}%")

    plot_results(model, device, scale, final_perimeter, perimeter_exact)


def plot_results(model, device, scale, calc_p, true_p):
    plot_config = config['plot']
    x = np.linspace(plot_config['x_min'], plot_config['x_max'], plot_config['grid_points'])
    y = np.linspace(plot_config['y_min'], plot_config['y_max'], plot_config['grid_points'])
    X, Y = np.meshgrid(x, y)

    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
    pts.requires_grad_(True)

    phi = model(pts)
    div_phi = get_divergence(phi, pts)

    phi_np = phi.detach().cpu().numpy()
    div_np = div_phi.detach().cpu().numpy()

    U = phi_np[:, 0].reshape(X.shape)
    V = phi_np[:, 1].reshape(X.shape)
    D = div_np.reshape(X.shape)

    mask = is_inside_lemniscate(
        torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32),
        scale,
    ).numpy().reshape(X.shape)
    D[~mask] = np.nan

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), cmap='autumn', density=1.5)
    ax[0].set_title('Vector Field $\\phi$')

    bx_r, by_r, bx_l, by_l = generate_boundary(scale)
    ax[0].plot(bx_r, by_r, 'b--', linewidth=2, label='Lemniscate')
    ax[0].plot(bx_l, by_l, 'b--', linewidth=2)
    ax[0].legend()
    ax[0].set_aspect('equal')

    im = ax[1].contourf(X, Y, D, levels=30, cmap='RdBu_r')
    ax[1].set_title(f'Divergence $\\nabla \\cdot \\phi$\\nCalc: {calc_p:.3f} | True: {true_p:.3f}')
    ax[1].plot(bx_r, by_r, 'b--', linewidth=2)
    ax[1].plot(bx_l, by_l, 'b--', linewidth=2)
    ax[1].set_aspect('equal')
    fig.colorbar(im, ax=ax[1])

    plt.tight_layout()
    plt.savefig(plot_config['output_file'], dpi=150)
    print(f"Plot saved to {plot_config['output_file']}")


if __name__ == '__main__':
    train()
