#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import time

# Set seed for reproducibility (or randomness)
# Use current time as seed for randomness if desired, or set a fixed integer
seed = int(time.time())
print(f"Random Seed: {seed}")
torch.manual_seed(seed)
np.random.seed(seed)

class VectorFieldNetwork(nn.Module):
    def __init__(self, hidden_dim=128, layers=4):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.output_layer = nn.Linear(hidden_dim, 2)
        # Using Tanh for smooth activation
        self.activation = nn.Tanh() 

    def forward(self, x):
        out = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        v = self.output_layer(out)
        
        # Soft-constraint enforcement: |phi| <= 1
        # Similar to the original code, we scale v to be within the unit ball
        v_norm = torch.norm(v, dim=1, keepdim=True)
        scale = torch.tanh(v_norm) / (v_norm + 1e-6)
        phi = v * scale
        return phi

def get_divergence(phi, x):
    # Compute divergence of phi with respect to x using autograd
    phi_x = phi[:, 0]
    phi_y = phi[:, 1]
    
    grad_x = torch.autograd.grad(phi_x, x, torch.ones_like(phi_x), create_graph=True)[0][:, 0]
    grad_y = torch.autograd.grad(phi_y, x, torch.ones_like(phi_y), create_graph=True)[0][:, 1]
    
    return grad_x + grad_y

def is_inside_cardioid(pts):
    """
    Check if points are inside the Cardioid defined by r = 1 + cos(theta).
    Cartesian eq: (x^2 + y^2 - x)^2 < x^2 + y^2
    """
    x = pts[:, 0]
    y = pts[:, 1]
    
    term1 = (x**2 + y**2 - x)**2
    term2 = x**2 + y**2
    return term1 < term2

def evaluate_perimeter(model, device, area_exact, n_samples=500000):
    model.eval()
    
    # Generate points in bounding box [-0.5, 2.5] x [-1.5, 1.5]
    # Width = 3.0, Height = 3.0. Area_box = 9.0
    x_min, x_max = -0.5, 2.5
    y_min, y_max = -1.5, 1.5
    
    x_raw = torch.rand(n_samples * 2, 2, device=device)
    x_raw[:, 0] = x_raw[:, 0] * (x_max - x_min) + x_min
    x_raw[:, 1] = x_raw[:, 1] * (y_max - y_min) + y_min
    
    mask = is_inside_cardioid(x_raw)
    x_in = x_raw[mask][:n_samples]
    
    if len(x_in) < n_samples:
        print(f"Warning: requested {n_samples} samples but only found {len(x_in)} inside. Accuracy may suffer.")
    
    batch_size = 5000 
    total_div = 0.0
    num_batches = 0
    
    for i in range(0, len(x_in), batch_size):
        batch = x_in[i:i+batch_size].clone().detach()
        batch.requires_grad_(True)
        
        phi = model(batch)
        div = get_divergence(phi, batch)
        
        total_div += torch.sum(div).item()
        num_batches += len(batch)
        
    avg_div = total_div / num_batches
    # Formula: Perimeter = Integral(div phi) / Area_Indicator ???
    # No, Divergence Theorem: Integral_Vol(div phi) = Integral_Surf(phi . n).
    # If phi = n on boundary, then Integral_Surf(1) = Perimeter.
    # So Integral_Vol(div phi) approx Perimeter.
    # VPINN formulation: Loss = - Integral(div phi * Indicator). 
    # The integral is approximated by Monte Carlo:
    # Integral ~ Volume * Mean(div phi)
    # Here Volume is the Area of the Cardioid.
    
    perimeter = avg_div * area_exact
    return perimeter

def train():
    # Setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA acceleration")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
    model = VectorFieldNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    iterations = 5000 # A bit more iterations for complex shape
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    
    batch_size = 8192
    
    # Exact values for Cardioid r = 1 + cos(theta)
    # Area = 3/2 * pi
    # Perimeter = 8
    AREA_EXACT = 1.5 * np.pi
    PERIMETER_EXACT = 8.0
    
    print(f"Target: Cardioid (r = 1 + cos(theta))")
    print(f"Exact Area: {AREA_EXACT:.5f}")
    print(f"Exact Perimeter: {PERIMETER_EXACT:.5f}")
    
    loss_history = []
    
    print("Starting training...")
    
    try:
        for i in range(iterations):
            model.train()
            
            # Sampling in Bounding Box
            x_min, x_max = -0.6, 2.6
            y_min, y_max = -1.6, 1.6
            
            x_raw = torch.rand(batch_size * 3, 2, device=device) # Oversample
            x_raw[:, 0] = x_raw[:, 0] * (x_max - x_min) + x_min
            x_raw[:, 1] = x_raw[:, 1] * (y_max - y_min) + y_min
            
            mask = is_inside_cardioid(x_raw)
            x_in = x_raw[mask][:batch_size]
            
            if len(x_in) < batch_size:
                continue
                
            x_in.requires_grad_(True)
            phi = model(x_in)
            div_phi = get_divergence(phi, x_in)
            
            # Maximize Integral(div phi). Loss = - Mean(div phi) * Area_Exact
            loss = - torch.mean(div_phi) * AREA_EXACT
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            loss_history.append(loss.item())
            
            if i % 500 == 0:
                print(f"Iter {i}: Loss = {loss.item():.5f} | Est. P = {-loss.item():.5f}")
                
    except KeyboardInterrupt:
        print("\nInterrupted.")

    # Evaluation
    final_perimeter = evaluate_perimeter(model, device, AREA_EXACT)
    print(f"\nFinal Results:")
    print(f"Calculated Perimeter: {final_perimeter:.5f}")
    print(f"Analytical Perimeter: {PERIMETER_EXACT:.5f}")
    print(f"Absolute Error: {abs(final_perimeter - PERIMETER_EXACT):.5f}")
    print(f"Relative Error: {abs(final_perimeter - PERIMETER_EXACT)/PERIMETER_EXACT*100:.2f}%")
    
    plot_results(model, device, final_perimeter, PERIMETER_EXACT)

def plot_results(model, device, calc_p, true_p):
    # Grid for plotting
    x = np.linspace(-1.0, 3.0, 200)
    y = np.linspace(-2.0, 2.0, 200)
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
    
    # Mask outside points for cleaner visualization
    mask = ( (X**2 + Y**2 - X)**2 < (X**2 + Y**2) )
    D[~mask] = np.nan
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Vector Field
    strm = ax[0].streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), cmap='autumn', density=1.5)
    ax[0].set_title('Vector Field $\phi$')
    
    # Draw Boundary
    theta = np.linspace(0, 2*np.pi, 500)
    r = 1 + np.cos(theta)
    bx = r * np.cos(theta)
    by = r * np.sin(theta)
    ax[0].plot(bx, by, 'b--', linewidth=2, label='Cardioid')
    ax[0].legend()
    ax[0].set_aspect('equal')

    # Divergence
    im = ax[1].contourf(X, Y, D, levels=30, cmap='RdBu_r')
    ax[1].set_title(f'Divergence $\nabla \cdot \phi$\nCalc: {calc_p:.3f} | True: {true_p:.3f}')
    ax[1].plot(bx, by, 'b--', linewidth=2)
    ax[1].set_aspect('equal')
    fig.colorbar(im, ax=ax[1])
    
    plt.tight_layout()
    plt.savefig('complex_perimeter_result.png', dpi=150)
    print("Plot saved to complex_perimeter_result.png")

if __name__ == '__main__':
    train()
