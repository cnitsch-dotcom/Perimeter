#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

class VectorFieldNetwork(nn.Module):
    """
    Critic: Approximates the field phi such that |phi| <= 1.
    Maximizes Integral_Omega (div phi) which equals Perimeter(Omega).
    """
    def __init__(self, hidden_dim=64, layers=3):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.activation = nn.SiLU() # Smooth activation for valid gradients 


    def forward(self, x):
        out = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        v = self.output_layer(out)
        
        # Soft-constraint enforcement: |phi| <= 1
        # Reverted back to Tanh as requested by user.
        v_norm = torch.norm(v, dim=1, keepdim=True)
        scale = torch.tanh(v_norm) / (v_norm + 1e-6)
        phi = v * scale
        return phi

class ShapeNetwork(nn.Module):
    """
    Generator: Defines the shape Omega by outputting a characteristic function u(x) in [0, 1].
    Minimizes J = Perimeter + lambda / Area.
    u(x) -> 1 inside, 0 outside.
    """
    def __init__(self, hidden_dim=64, layers=3):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()
        self.output_activation = nn.Sigmoid()

    def forward(self, x, return_prob=False):
        out = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        raw_u = self.output_activation(self.output_layer(out) * 5.0) # Temperature scaling for sharper sigmoid
        
        # Apply soft boundary mask to force u -> 0 at edges of [-1.5, 1.5]
        # x is in [-1.5, 1.5]. 
        # Mask = Sigmoid(slope * (1.4 - |x|))
        slope = 3.0
        abs_x = torch.abs(x)
        mask_x = torch.sigmoid(slope * (1.3 - abs_x[:, 0:1])) # Slightly smaller box 1.3
        mask_y = torch.sigmoid(slope * (1.3 - abs_x[:, 1:2]))
        
        # Straight-Through Estimator (STE)
        # Forward: u = 1 if prob > 0.5 else 0
        # Backward: Gradient flows through probabilities (raw_u)
        
        # Apply mask to probabilities first
        prob = raw_u * mask_x * mask_y
        
        # Hard threshold
        u_hard = (prob > 0.5).float()
        
        # STE Trick: u = prob + (u_hard - prob).detach()
        # In forward: prob cancels out -> result is u_hard
        # In backward: (u_hard - prob).detach() has 0 grad -> grad is grad(prob)
        u = prob + (u_hard - prob).detach()
        
        if return_prob:
            return prob
            
        return u

def get_divergence(phi, x):
    # Compute divergence of phi with respect to x using autograd
    phi_x = phi[:, 0]
    phi_y = phi[:, 1]
    
    grad_x = torch.autograd.grad(phi_x, x, torch.ones_like(phi_x), create_graph=True)[0][:, 0]
    grad_y = torch.autograd.grad(phi_y, x, torch.ones_like(phi_y), create_graph=True)[0][:, 1]
    
    return grad_x + grad_y

def train():
    # Setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    LAMBDA = 4.0 # Weight for Area term (Minimize P + lambda/A)
    # Theoretical optimal shape is a circle. 
    # Radius R minimizes 2*pi*R + lambda/(pi*R^2).
    # d/dR = 2*pi - 2*lambda/(pi*R^3) = 0 => R^3 = lambda/pi^2 => R = (lambda/pi^2)^(1/3)
    target_R = (LAMBDA / (np.pi**2))**(1/3)
    print(f"Target Radius for Lambda={LAMBDA}: {target_R:.4f}")

    pretrain_radius = 0.8

    critic = VectorFieldNetwork(hidden_dim=128, layers=4).to(device)
    generator = ShapeNetwork(hidden_dim=128, layers=4).to(device)
    
    # Optimizers
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3) # Standard rate
    opt_gen = optim.Adam(generator.parameters(), lr=2e-4) # Slow generator
    
    iterations = 200
    batch_size = 2048
    
    print("Starting GAN training...")
    
    try:
        for i in range(iterations):
            # Domain: [-1.5, 1.5] x [-1.5, 1.5]
            x_raw = (torch.rand(batch_size, 2, device=device) - 0.5) * 3.0
            x_raw.requires_grad_(True)
            
            # --- 1. Train Critic (Maximize Perimeter Estimation) ---
            # We want to Maximize Integral(u * div phi).
            # Loss_Critic = - Mean(u.detach() * div phi)
            
            # Pre-training Phase: For first few iterations, do not update generator.
            # And use a fixed, perfect circle shape to guide the Critic?
            # Or just let it learn on the initial generator shape (which is small blob).
            # Let's force a fixed circle shape for pre-training to bootstrap the source field.
            
            # Critic needs more steps to approximate perimeter reliably.
            current_critic_steps = 800
            generator_steps = 3
            
            for _ in range(current_critic_steps):
                x_c = (torch.rand(batch_size, 2, device=device) - 0.5) * 3.0
                x_c.requires_grad_(True)
                
                if i < 30:
                    # Synthetic Circle for Pre-training: Radius pretrain_radius
                    # u_fixed = Sigmoid(10 * (pretrain_radius - r))
                    r_c = torch.norm(x_c, dim=1, keepdim=True)
                    # Use strict binary step function for pre-training
                    u_fixed = (r_c < pretrain_radius).float()
                else:
                    u_fixed = generator(x_c).detach()
                    
                phi = critic(x_c)
                div_phi = get_divergence(phi, x_c)
                
                loss_critic = - torch.mean(u_fixed * div_phi) * 9.0
                
                opt_critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()
            
            # --- 2. Train Generator (Optimize Shape) ---
            if i >= 30:
                for _ in range(generator_steps):
                    x_raw = (torch.rand(batch_size, 2, device=device) - 0.5) * 3.0
                    x_raw.requires_grad_(True)
                    u = generator(x_raw)
                    
                    phi_for_div = critic(x_raw) 
                    div_phi = get_divergence(phi_for_div, x_raw)
                    div_val = div_phi.detach()
                    
                    area_val = torch.mean(u) * 9.0
                    perimeter_val = torch.mean(u * div_val) * 9.0
                    
                    # Minimizing P + Lambda/A
                    # Add barrier for A -> 0?
                    term_area = LAMBDA / (area_val + 1e-6)
                    
                    # Binary Penalty removed (STE guarantees binary output)
                    
                    loss_gen = perimeter_val + term_area
                    
                    opt_gen.zero_grad()
                    loss_gen.backward()
                    opt_gen.step()
            else:
                # Just for logging
                # We need gradients for get_divergence even if we don't opt step
                u = generator(x_raw).detach()
                
                phi_for_div = critic(x_raw)
                div_val = get_divergence(phi_for_div, x_raw).detach()
                
                area_val = torch.mean(u) * 9.0
                perimeter_val = torch.mean(u * div_val) * 9.0
                term_area = LAMBDA / (area_val + 1e-6)
                loss_gen = perimeter_val + term_area
            
            if i % 10 == 0:
                plot_results(generator, critic, device, target_R, step=f"iter_{i}")
            
            if i < 30 and i % 10 == 0:
                # Visualize the fixed pretraining shape and critic field
                plot_results(generator, critic, device, target_R, step=f"pretrain_{i}", fixed_radius=pretrain_radius)

            if i % 10 == 0:
                # Check metrics
                phi_norm = torch.norm(phi_for_div, dim=1).mean().item()
                p_min_theoretical = 2 * np.sqrt(np.pi * area_val.item())
                
                # Binary_Err should be 0 by definition now
                # Verify for user confidence:
                is_binary = torch.all((u == 0) | (u == 1))
                if not is_binary:
                    print("WARNING: u is NOT binary!")
                
                print(f"Iter {i}: L_Crit={loss_critic.item():.4f} | L_Gen={loss_gen.item():.4f} | P_est={perimeter_val.item():.4f} | A_est={area_val.item():.4f} | P_min_th={p_min_theoretical:.4f}", flush=True)
                
                if i < 30:
                    p_expected = 2 * np.pi * pretrain_radius
                    p_estimated = -loss_critic.item()  # loss_critic = -mean(u_fixed*div_phi)*9
                    print(f"   Pretrain circle r={pretrain_radius}: P_expected={p_expected:.4f} | P_estimated_by_critic={p_estimated:.4f}")

    except KeyboardInterrupt:
        print("Interrupted.")
        
    print("Training finished.")
    plot_results(generator, critic, device, target_R)

def plot_results(generator, critic, device, target_R, step=None, fixed_radius=None):
    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)
    
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device)
    pts.requires_grad_(True)
    
    if fixed_radius is not None:
        r_grid = torch.sqrt(torch.sum(pts**2, dim=1, keepdim=True))
        u_prob = (r_grid < fixed_radius).float()
        u_binary = u_prob
    else:
        u_prob = generator(pts, return_prob=True) # Get probabilities for first plot
        u_binary = generator(pts) # Get binary output for second plot and divergence
    phi = critic(pts)
    div = get_divergence(phi, pts)
    
    u_prob_np = u_prob.detach().cpu().numpy().reshape(X.shape)
    u_binary_np = u_binary.detach().cpu().numpy().reshape(X.shape)
    phi_np = phi.detach().cpu().numpy()
    div_np = div.detach().cpu().numpy().reshape(X.shape)
    
    U = phi_np[:, 0].reshape(X.shape)
    V = phi_np[:, 1].reshape(X.shape)
    
    fig, ax = plt.subplots(1, 4, figsize=(24, 6)) # Added 4th plot for Binary Set
    
    # 1. Continuous u(x) - Probabilities
    im1 = ax[0].contourf(X, Y, u_prob_np, levels=20, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title('Generator Prob $P(u=1)$')
    ax[0].set_aspect('equal')
    ax[0].set_xlim(-1.5, 1.5)
    ax[0].set_ylim(-1.5, 1.5)
    fig.colorbar(im1, ax=ax[0])
    
    # 2. Binary Set (Thresholded)
    im2 = ax[1].imshow(u_binary_np, extent=[-1.5, 1.5, -1.5, 1.5], origin='lower', cmap='binary', vmin=0, vmax=1)
    ax[1].set_title('Binary Set $\Omega$')
    ax[1].set_aspect('equal')
    circle = plt.Circle((0, 0), target_R, color='r', fill=False, linestyle='--', linewidth=2, label='Theory')
    ax[1].add_patch(circle)
    ax[1].legend()
    
    # 3. Vector Field
    norm_mag = np.sqrt(U**2 + V**2)
    strm = ax[2].streamplot(X, Y, U, V, color=norm_mag, cmap='autumn', density=1.0, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    ax[2].set_title('Critic Field $\phi$')
    ax[2].set_aspect('equal')
    ax[2].set_xlim(-1.5, 1.5)
    ax[2].set_ylim(-1.5, 1.5)
    ax[2].add_patch(plt.Circle((0, 0), target_R, color='b', fill=False, linestyle='--', linewidth=2))
    fig.colorbar(strm.lines, ax=ax[2], label='$|\phi|$', boundaries=np.linspace(0.0, 1.0, 6))
    
    # 4. Divergence
    im4 = ax[3].contourf(X, Y, div_np, levels=20, cmap='RdBu_r')
    ax[3].set_title('Divergence $\nabla \cdot \phi$')
    ax[3].set_aspect('equal')
    ax[3].set_xlim(-1.5, 1.5)
    ax[3].set_ylim(-1.5, 1.5)
    fig.colorbar(im4, ax=ax[3])
    ax[3].add_patch(plt.Circle((0, 0), target_R, color='b', fill=False, linestyle='--', linewidth=2))
    
    plt.tight_layout()
    outfile = "shape_optimization_result.png" if step is None else f"shape_optimization_result_{step}.png"
    plt.savefig(outfile)
    print(f"Saved result to {outfile}")

if __name__ == '__main__':
    train()
