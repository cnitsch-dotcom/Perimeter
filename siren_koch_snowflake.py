#!/usr/bin/env python3

import json
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

CONFIG_PATH = Path(__file__).with_name('koch_snowflake_config.json')


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


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        w0: float = 30.0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.is_first = bool(is_first)
        self.w0 = float(w0)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_siren_weights()

    def _init_siren_weights(self):
        if self.is_first:
            bound = 1.0 / self.in_features
        else:
            bound = math.sqrt(6.0 / self.in_features) / self.w0

        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class VectorFieldNetwork(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        layers: int = 3,
        w0_first: float = 30.0,
        w0_hidden: float = 1.0,
    ):
        super().__init__()
        self.input_layer = SineLayer(2, hidden_dim, is_first=True, w0=w0_first)
        self.hidden_layers = nn.ModuleList(
            SineLayer(hidden_dim, hidden_dim, is_first=False, w0=w0_hidden) for _ in range(layers)
        )
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
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


def build_initial_triangle(side_length):
    height = side_length * np.sqrt(3.0) / 2.0
    vertices = np.array(
        [
            [0.0, 0.0],
            [side_length, 0.0],
            [side_length / 2.0, height],
        ],
        dtype=np.float32,
    )
    centroid = np.mean(vertices, axis=0)
    return vertices - centroid


def koch_refine(vertices):
    new_vertices = []
    rotation = np.exp(-1j * np.pi / 3.0)
    n_vertices = len(vertices)

    for i in range(n_vertices):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_vertices]
        z1 = complex(p1[0], p1[1])
        z2 = complex(p2[0], p2[1])
        dz = z2 - z1

        za = z1 + dz / 3.0
        zb = z1 + 2.0 * dz / 3.0
        zc = za + (dz / 3.0) * rotation

        new_vertices.append([z1.real, z1.imag])
        new_vertices.append([za.real, za.imag])
        new_vertices.append([zc.real, zc.imag])
        new_vertices.append([zb.real, zb.imag])

    return np.array(new_vertices, dtype=np.float32)


def generate_koch_snowflake(side_length, iteration):
    vertices = build_initial_triangle(side_length)
    for _ in range(iteration):
        vertices = koch_refine(vertices)
    return vertices


def polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def analytical_perimeter(side_length, iteration):
    return 3.0 * side_length * (4.0 / 3.0) ** iteration


def analytical_area(side_length, iteration):
    a0 = (np.sqrt(3.0) / 4.0) * side_length**2
    return a0 * (8.0 / 5.0 - (3.0 / 5.0) * (4.0 / 9.0) ** iteration)


def is_inside_polygon(pts, polygon):
    x = pts[:, 0]
    y = pts[:, 1]
    inside = torch.zeros_like(x, dtype=torch.bool)

    n_vertices = polygon.shape[0]
    for i in range(n_vertices):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n_vertices]
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        intersects = ((y1 > y) != (y2 > y)) & (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        inside = torch.logical_xor(inside, intersects)

    return inside


def sample_points_inside(device, n_points, oversample_factor, bbox, polygon_t):
    x_min = bbox['x_min']
    x_max = bbox['x_max']
    y_min = bbox['y_min']
    y_max = bbox['y_max']

    target = n_points
    collected = []
    total_inside = 0
    max_rounds = 20

    for _ in range(max_rounds):
        n_raw = max(target * oversample_factor, target)
        x_raw = torch.rand(n_raw, 2, device=device)
        x_raw[:, 0] = x_raw[:, 0] * (x_max - x_min) + x_min
        x_raw[:, 1] = x_raw[:, 1] * (y_max - y_min) + y_min

        mask = is_inside_polygon(x_raw, polygon_t)
        x_in = x_raw[mask]
        if len(x_in) > 0:
            collected.append(x_in)
            total_inside += len(x_in)

        if total_inside >= target:
            break

    if total_inside == 0:
        return torch.empty(0, 2, device=device)

    points = torch.cat(collected, dim=0)
    return points[:target]


def estimate_area_mc(device, n_raw, bbox, polygon_t):
    x_min = bbox['x_min']
    x_max = bbox['x_max']
    y_min = bbox['y_min']
    y_max = bbox['y_max']
    area_box = (x_max - x_min) * (y_max - y_min)

    pts = torch.rand(n_raw, 2, device=device)
    pts[:, 0] = pts[:, 0] * (x_max - x_min) + x_min
    pts[:, 1] = pts[:, 1] * (y_max - y_min) + y_min

    mask = is_inside_polygon(pts, polygon_t)
    return mask.float().mean().item() * area_box


def plot_sampling_points(device, polygon_np, polygon_t, bbox, n_raw, output_file):
    x_min = bbox['x_min']
    x_max = bbox['x_max']
    y_min = bbox['y_min']
    y_max = bbox['y_max']

    pts = torch.rand(n_raw, 2, device=device)
    pts[:, 0] = pts[:, 0] * (x_max - x_min) + x_min
    pts[:, 1] = pts[:, 1] * (y_max - y_min) + y_min

    with torch.no_grad():
        mask = is_inside_polygon(pts, polygon_t).detach().cpu().numpy()
    pts_np = pts.detach().cpu().numpy()

    closed = np.vstack([polygon_np, polygon_np[0]])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        pts_np[~mask, 0],
        pts_np[~mask, 1],
        s=1,
        c='#d62728',
        alpha=0.25,
        linewidths=0,
        label='outside',
    )
    ax.scatter(
        pts_np[mask, 0],
        pts_np[mask, 1],
        s=1,
        c='#2ca02c',
        alpha=0.35,
        linewidths=0,
        label='inside',
    )
    ax.plot(closed[:, 0], closed[:, 1], 'k-', linewidth=1.2, label='boundary')
    ax.set_title(f"Monte Carlo sampling (raw={n_raw}, inside={int(mask.sum())})")
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Debug sampling plot saved to {output_file}")


def plot_training_points(x_in, polygon_np, bbox, output_file):
    x_min = bbox['x_min']
    x_max = bbox['x_max']
    y_min = bbox['y_min']
    y_max = bbox['y_max']

    pts_np = x_in.detach().cpu().numpy()
    closed = np.vstack([polygon_np, polygon_np[0]])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        pts_np[:, 0],
        pts_np[:, 1],
        s=1,
        c='#1f77b4',
        alpha=0.5,
        linewidths=0,
        label=f'x_in (N={len(pts_np)})',
    )
    ax.plot(closed[:, 0], closed[:, 1], 'k-', linewidth=1.2, label='boundary')
    ax.set_title("Training points used for loss (x_in)")
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Debug training points plot saved to {output_file}")


def evaluate_perimeter(model, device, polygon_t, eval_config):
    model.eval()

    n_samples = eval_config['n_samples']
    batch_size = eval_config['batch_size']
    bbox = eval_config['bounding_box']

    area_estimate = estimate_area_mc(device, eval_config['area_samples'], bbox, polygon_t)
    x_in = sample_points_inside(device, n_samples, eval_config['oversample_factor'], bbox, polygon_t)

    if len(x_in) < n_samples:
        print(f"Warning: requested {n_samples} samples but found {len(x_in)} inside. Accuracy may suffer.")

    total_div = 0.0
    num_points = 0

    for i in range(0, len(x_in), batch_size):
        batch = x_in[i:i + batch_size].clone().detach()
        batch.requires_grad_(True)

        phi = model(batch)
        div = get_divergence(phi, batch)

        total_div += torch.sum(div).item()
        num_points += len(batch)

    if num_points == 0:
        return 0.0, area_estimate

    avg_div = total_div / num_points
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

    side_length = float(shape_config['side_length'])
    iteration = int(shape_config['n'])

    if iteration < 0:
        raise ValueError("Il parametro n deve essere >= 0.")
    if iteration > shape_config['max_n']:
        raise ValueError(f"n troppo alto per la configurazione corrente. Usa n <= {shape_config['max_n']}.")

    boundary_np = generate_koch_snowflake(side_length, iteration)
    boundary_t = torch.tensor(boundary_np, dtype=torch.float32, device=device)

    area_exact = analytical_area(side_length, iteration)
    perimeter_exact = analytical_perimeter(side_length, iteration)
    area_polygon = polygon_area(boundary_np)

    w0_first = model_config.get("w0_first", 30.0)
    w0_hidden = model_config.get("w0_hidden", 1.0)

    model = VectorFieldNetwork(
        hidden_dim=model_config.get('hidden_dim', 256),
        layers=model_config.get('layers', 3),
        w0_first=w0_first,
        w0_hidden=w0_hidden,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    iterations = train_config['iterations']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    batch_size = train_config['batch_size']
    penalty_multiplier = penalty_config['multiplier']

    print("Target: Koch Snowflake (SIREN)")
    print(f"Side length s: {side_length:.6f}")
    print(f"Iteration n: {iteration}")
    print(f"Boundary vertices: {len(boundary_np)}")
    print(f"Analytical Area: {area_exact:.6f}")
    print(f"Polygon Area Check: {area_polygon:.6f}")
    print(f"Analytical Perimeter: {perimeter_exact:.6f}")
    print(f"Penalty Multiplier: {penalty_multiplier:.6f}")
    print(f"Model: hidden_dim={model_config.get('hidden_dim', 256)}, layers={model_config.get('layers', 3)}")
    print(f"SIREN: w0_first={w0_first}, w0_hidden={w0_hidden}")
    print("Starting training...")

    debug_config = config.get('debug', {})
    debug_plot_enabled = bool(debug_config.get('plot_sampling_points', False))
    debug_plot_training_enabled = bool(debug_config.get('plot_training_points', False))
    debug_plot_iteration = int(debug_config.get('iteration', 0))
    debug_plot_done = False
    debug_plot_training_done = False

    try:
        for i in range(iterations):
            model.train()

            x_in = sample_points_inside(
                device=device,
                n_points=batch_size,
                oversample_factor=train_config['oversample_factor'],
                bbox=train_config['bounding_box'],
                polygon_t=boundary_t,
            )

            if len(x_in) < batch_size:
                continue

            if debug_plot_enabled and (not debug_plot_done) and i == debug_plot_iteration:
                plot_sampling_points(
                    device=device,
                    polygon_np=boundary_np,
                    polygon_t=boundary_t,
                    bbox=train_config['bounding_box'],
                    n_raw=int(debug_config.get('n_raw', 200000)),
                    output_file=str(debug_config.get('output_file', 'koch_sampling_debug.png')),
                )
                debug_plot_done = True

            if debug_plot_training_enabled and (not debug_plot_training_done) and i == debug_plot_iteration:
                plot_training_points(
                    x_in=x_in,
                    polygon_np=boundary_np,
                    bbox=train_config['bounding_box'],
                    output_file=str(debug_config.get('training_output_file', 'koch_training_points_debug.png')),
                )
                debug_plot_training_done = True

            x_in.requires_grad_(True)
            phi = model(x_in)
            div_phi = get_divergence(phi, x_in)

            perimeter_loss = -torch.mean(div_phi) * area_exact
            divergence_penalty = torch.mean(div_phi**2)
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

    final_perimeter, estimated_area = evaluate_perimeter(
        model,
        device,
        boundary_t,
        config['evaluation'],
    )

    print("\nFinal Results:")
    print(f"Estimated Area: {estimated_area:.5f}")
    print(f"Analytical Area: {area_exact:.5f}")
    print(f"Area Absolute Error: {abs(estimated_area - area_exact):.5f}")
    print(f"Area Relative Error: {abs(estimated_area - area_exact) / area_exact * 100:.2f}%")
    print(f"Calculated Perimeter: {final_perimeter:.5f}")
    print(f"Analytical Perimeter: {perimeter_exact:.5f}")
    print(f"Absolute Error: {abs(final_perimeter - perimeter_exact):.5f}")
    print(f"Relative Error: {abs(final_perimeter - perimeter_exact) / perimeter_exact * 100:.2f}%")

    plot_results(model, device, boundary_np, boundary_t, final_perimeter, perimeter_exact)


def plot_results(model, device, boundary_np, boundary_t, calc_p, true_p):
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

    with torch.no_grad():
        mask = is_inside_polygon(pts.detach(), boundary_t).cpu().numpy().reshape(X.shape)
    D[~mask] = np.nan

    closed = np.vstack([boundary_np, boundary_np[0]])

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), cmap='autumn', density=1.5)
    ax[0].plot(closed[:, 0], closed[:, 1], 'b--', linewidth=1.6, label='Koch Snowflake')
    ax[0].set_title('Vector Field $\\phi$')
    ax[0].legend()
    ax[0].set_aspect('equal')

    im = ax[1].contourf(X, Y, D, levels=30, cmap='RdBu_r')
    ax[1].plot(closed[:, 0], closed[:, 1], 'b--', linewidth=1.6)
    ax[1].set_title(f'Divergence $\\nabla \\cdot \\phi$\\nCalc: {calc_p:.3f} | True: {true_p:.3f}')
    ax[1].set_aspect('equal')
    fig.colorbar(im, ax=ax[1])

    plt.tight_layout()
    plt.savefig(plot_config['output_file'], dpi=150)
    print(f"Plot saved to {plot_config['output_file']}")


if __name__ == '__main__':
    train()

