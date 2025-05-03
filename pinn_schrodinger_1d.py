# %%
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

# ===== Parameters =====
L = 1.0   # Well width
n = 1     # Quantum number
hbar = 1.0
m = 1.0
# Theoretical energy eigenvalue
E_true = (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
print(f"True energy eigenvalue: {E_true:.6f}")

# ===== Generate training data =====
# Boundary points for enforcing boundary conditions
num_boundary_points = 100
x_boundary = np.array([[0.0]]*num_boundary_points + [[L]]*num_boundary_points)
t_boundary = np.random.uniform(0, 1, size=(2*num_boundary_points, 1))
boundary_points = np.concatenate([x_boundary, t_boundary], axis=1)

# Interior points for PDE residual
num_interior_points = 10000
x_interior = np.random.uniform(0, L, size=(num_interior_points, 1))
t_interior = np.random.uniform(0, 1, size=(num_interior_points, 1))
interior_points = np.concatenate([x_interior, t_interior], axis=1)

# Initial condition points
num_initial_points = 200
x_initial = np.linspace(0, L, num_initial_points)[:, None]
t_initial = np.zeros_like(x_initial)
initial_points = np.concatenate([x_initial, t_initial], axis=1)
# Initial condition values (t=0)
initial_real = np.sqrt(2/L) * np.sin(n * np.pi * x_initial / L)
initial_imag = np.zeros_like(initial_real)
initial_values = np.concatenate([initial_real, initial_imag], axis=1)

# Data points for supervised learning
x_data = np.linspace(0, L, 100)[:, None]
t_data = np.linspace(0, 1, 50)[:, None]
Xg, Tg = np.meshgrid(x_data, t_data)
data_points = np.concatenate([Xg.reshape(-1,1), Tg.reshape(-1,1)], axis=1)
# Analytical solution (real and imaginary parts)
psi_real = np.sqrt(2/L) * np.sin(n * np.pi * Xg.reshape(-1,1) / L) * np.cos(E_true * Tg.reshape(-1,1) / hbar)
psi_imag = -np.sqrt(2/L) * np.sin(n * np.pi * Xg.reshape(-1,1) / L) * np.sin(E_true * Tg.reshape(-1,1) / hbar)
data_values = np.concatenate([psi_real, psi_imag], axis=1)

# Convert to torch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
boundary_points = torch.tensor(boundary_points, dtype=torch.float32, device=device)
interior_points = torch.tensor(interior_points, dtype=torch.float32, device=device)
initial_points = torch.tensor(initial_points, dtype=torch.float32, device=device)
initial_values = torch.tensor(initial_values, dtype=torch.float32, device=device)
data_points = torch.tensor(data_points, dtype=torch.float32, device=device)
data_values = torch.tensor(data_values, dtype=torch.float32, device=device)

print(f"Using device: {device}")
print(f"Number of training points - Boundary: {boundary_points.shape[0]}, Interior: {interior_points.shape[0]}, Initial: {initial_points.shape[0]}, Data: {data_points.shape[0]}")

# ===== Neural Network Definition =====
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Use Tanh activation for better handling of boundary conditions
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
        
        # Better initialization with smaller weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        return self.net(x)

net = PINN().to(device)

# Learnable energy parameter
E_pred = nn.Parameter(torch.tensor(E_true * 1.1, device=device))  # Start with initial guess

# ===== Training Configuration =====
num_epochs = 20000  # More epochs for complex problems
learning_rate = 1e-3

# Initialize optimizer - Adam works well for PINNs
optimizer = torch.optim.Adam([
    {'params': net.parameters()},
    {'params': [E_pred], 'lr': learning_rate * 0.1}  # Slower learning rate for energy parameter
], lr=learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1000, 
    min_lr=1e-6
)

# ===== Loss Function =====
def compute_pde_residual(model, x, t):
    """Compute the PDE residuals at specified points"""
    # Create input with gradient tracking
    points = torch.cat([x, t], dim=1)
    points.requires_grad_(True)
    
    # Forward pass
    uv = model(points)
    u = uv[:, 0:1]  # Real part
    v = uv[:, 1:2]  # Imaginary part
    
    # Compute derivatives
    # Second-order spatial derivatives
    du_dx = autograd.grad(u.sum(), points, create_graph=True)[0][:, 0:1]
    d2u_dx2 = autograd.grad(du_dx.sum(), points, create_graph=True)[0][:, 0:1]
    dv_dx = autograd.grad(v.sum(), points, create_graph=True)[0][:, 0:1]
    d2v_dx2 = autograd.grad(dv_dx.sum(), points, create_graph=True)[0][:, 0:1]
    
    # Time derivatives
    du_dt = autograd.grad(u.sum(), points, create_graph=True)[0][:, 1:2]
    dv_dt = autograd.grad(v.sum(), points, create_graph=True)[0][:, 1:2]
    
    # PDE residuals (Schrödinger equation)
    res_u = hbar * du_dt + (hbar**2/(2*m)) * d2v_dx2
    res_v = hbar * dv_dt - (hbar**2/(2*m)) * d2u_dx2
    
    return res_u, res_v

def compute_loss():
    """Compute all loss components"""
    # 1. Data loss
    pred_data = net(data_points)
    data_loss = torch.mean((pred_data - data_values)**2)
    
    # 2. Initial condition loss
    pred_initial = net(initial_points)
    initial_loss = torch.mean((pred_initial - initial_values)**2)
    
    # 3. Boundary condition loss (psi = 0 at x=0 and x=L)
    pred_boundary = net(boundary_points)
    boundary_loss = torch.mean(pred_boundary**2)
    
    # 4. PDE residual loss at interior points
    x_interior = interior_points[:, 0:1]
    t_interior = interior_points[:, 1:2]
    res_u, res_v = compute_pde_residual(net, x_interior, t_interior)
    pde_loss = torch.mean(res_u**2 + res_v**2)
    
    # 5. Energy conservation loss - Eigenvalue problem at t=0
    x_eigen = initial_points[:, 0:1]
    t_eigen = initial_points[:, 1:2]
    pred_eigen = net(initial_points)
    u_eigen = pred_eigen[:, 0:1]  # Real part at t=0
    v_eigen = pred_eigen[:, 1:2]  # Imag part at t=0 (should be ~0)
    
    # Second derivatives for energy calculation
    points_eigen = torch.cat([x_eigen, t_eigen], dim=1)
    points_eigen.requires_grad_(True)
    u_eigen_r = net(points_eigen)[:, 0:1]
    du_dx = autograd.grad(u_eigen_r.sum(), points_eigen, create_graph=True)[0][:, 0:1]
    d2u_dx2 = autograd.grad(du_dx.sum(), points_eigen, create_graph=True)[0][:, 0:1]
    
    # Time-independent Schrödinger equation: -ħ²/(2m) ∇²ψ = E ψ
    eigen_loss = torch.mean(((-hbar**2/(2*m)) * d2u_dx2 - E_pred * u_eigen)**2)
    
    # Combine all losses with appropriate weights
    total_loss = data_loss + 10.0 * initial_loss + 10.0 * boundary_loss + 0.1 * pde_loss + 1.0 * eigen_loss
    
    return total_loss, data_loss, initial_loss, boundary_loss, pde_loss, eigen_loss

# %%
# ===== Training Loop =====
torch.cuda.empty_cache()  # Clear CUDA cache before training
best_loss = float('inf')
loss_history = []

# Use a progress bar for the epochs
progress_bar = tqdm(range(1, num_epochs+1), desc="Training")

for epoch in progress_bar:
    # Forward and backward pass
    optimizer.zero_grad()
    total_loss, data_loss, initial_loss, boundary_loss, pde_loss, eigen_loss = compute_loss()
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    
    # Update parameters
    optimizer.step()
    scheduler.step(total_loss)
    
    # Record loss
    loss_history.append([total_loss.item(), data_loss.item(), initial_loss.item(), 
                        boundary_loss.item(), pde_loss.item(), eigen_loss.item(), E_pred.item()])
    
    # Update progress bar
    progress_bar.set_postfix({
        'loss': f'{total_loss.item():.3e}',
        'E_pred': f'{E_pred.item():.3f}',
        'E_true': f'{E_true:.3f}'
    })
    
    # Save the best model
    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        best_model_state = {
            'model': net.state_dict(),
            'E_pred': E_pred.item()
        }
    
    # Print progress and visualize current solution periodically
    if epoch % 500 == 0 or epoch == 1:
        print(f"\nEpoch {epoch}")
        print(f"Loss: {total_loss.item():.4e}, Data: {data_loss.item():.4e}, Initial: {initial_loss.item():.4e}")
        print(f"Boundary: {boundary_loss.item():.4e}, PDE: {pde_loss.item():.4e}, Eigen: {eigen_loss.item():.4e}")
        print(f"E_pred: {E_pred.item():.6f}, E_true: {E_true:.6f}, Relative error: {100*abs(E_pred.item()-E_true)/E_true:.4f}%")
        
        # Visualize current solution
        with torch.no_grad():
            x_vis = torch.linspace(0, L, 200, device=device).reshape(-1, 1)
            t_vis = torch.zeros_like(x_vis)
            points_vis = torch.cat([x_vis, t_vis], dim=1)
            psi_vis = net(points_vis).cpu().numpy()
            psi_theory = np.sqrt(2/L) * np.sin(n * np.pi * x_vis.cpu().numpy() / L)
            
            plt.figure(figsize=(10, 5))
            plt.plot(x_vis.cpu().numpy(), psi_vis[:, 0], 'r-', label='PINN Real')
            plt.plot(x_vis.cpu().numpy(), psi_vis[:, 1], 'g-', label='PINN Imag')
            plt.plot(x_vis.cpu().numpy(), psi_theory, 'b--', label='Theory')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.title(f'Wavefunction at t=0 (Epoch {epoch})')
            plt.draw()
            plt.pause(0.1)
            plt.close()

# Load the best model
net.load_state_dict(best_model_state['model'])
E_pred.data = torch.tensor(best_model_state['E_pred'], device=device)

# %%
# ===== Results Visualization =====
loss_history = np.array(loss_history)
plt.figure(figsize=(15, 12))

# Plot loss curves
plt.subplot(2, 2, 1)
plt.semilogy(loss_history[:, 0], 'k-', label='Total Loss')
plt.semilogy(loss_history[:, 1], 'b-', label='Data Loss')
plt.semilogy(loss_history[:, 2], 'r-', label='Initial Loss')
plt.semilogy(loss_history[:, 3], 'g-', label='Boundary Loss')
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.legend()
plt.title('Training Losses (Part 1)')

plt.subplot(2, 2, 2)
plt.semilogy(loss_history[:, 4], 'c-', label='PDE Loss')
plt.semilogy(loss_history[:, 5], 'm-', label='Eigen Loss')
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.legend()
plt.title('Training Losses (Part 2)')

# Plot energy convergence
plt.subplot(2, 2, 3)
plt.plot(loss_history[:, 6], 'b-', label='Predicted Energy')
plt.axhline(y=E_true, color='r', linestyle='--', label=f'True Energy: {E_true:.4f}')
plt.grid(True, alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('Energy')
plt.legend()
plt.title('Energy Parameter Convergence')

# Visualize the final wavefunction at t=0
plt.subplot(2, 2, 4)
with torch.no_grad():
    x_test = torch.linspace(0, L, 200, device=device).reshape(-1, 1)
    t_test = torch.zeros_like(x_test)
    points_test = torch.cat([x_test, t_test], dim=1)
    psi_pred = net(points_test).cpu().numpy()
    psi_real_pred = psi_pred[:, 0]
    psi_imag_pred = psi_pred[:, 1]
    psi_theory = np.sqrt(2/L) * np.sin(n * np.pi * x_test.cpu().numpy() / L)

plt.plot(x_test.cpu().numpy(), psi_real_pred, 'r-', linewidth=2, label='PINN Real')
plt.plot(x_test.cpu().numpy(), psi_imag_pred, 'g-', linewidth=2, label='PINN Imag')
plt.plot(x_test.cpu().numpy(), psi_theory, 'b--', linewidth=1.5, label='Theory')
plt.xlabel('x')
plt.ylabel('$\psi(x)$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Wavefunction at t=0')

plt.tight_layout()
plt.savefig("pinn_schrodinger_1d_training.pdf", transparent=True)
plt.show()

# Visualize the wavefunction at different times
plt.figure(figsize=(12, 8))
times = [0.0, 0.25, 0.5, 0.75, 1.0]
for i, t_val in enumerate(times):
    plt.subplot(2, 3, i+1)
    with torch.no_grad():
        x_test = torch.linspace(0, L, 200, device=device).reshape(-1, 1)
        t_test = torch.ones_like(x_test) * t_val
        points_test = torch.cat([x_test, t_test], dim=1)
        psi_pred = net(points_test).cpu().numpy()
        psi_real_pred = psi_pred[:, 0]
        psi_imag_pred = psi_pred[:, 1]
        # Analytical solution
        psi_real_true = np.sqrt(2/L) * np.sin(n * np.pi * x_test.cpu().numpy() / L) * np.cos(E_true * t_val / hbar)
        psi_imag_true = -np.sqrt(2/L) * np.sin(n * np.pi * x_test.cpu().numpy() / L) * np.sin(E_true * t_val / hbar)
    
    plt.plot(x_test.cpu().numpy(), psi_real_pred, 'r-', label='PINN Real')
    plt.plot(x_test.cpu().numpy(), psi_imag_pred, 'g-', label='PINN Imag')
    plt.plot(x_test.cpu().numpy(), psi_real_true, 'r--', alpha=0.7, label='True Real')
    plt.plot(x_test.cpu().numpy(), psi_imag_true, 'g--', alpha=0.7, label='True Imag')
    plt.grid(True, alpha=0.3)
    plt.title(f't = {t_val}')
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig("pinn_schrodinger_1d_time_evolution.pdf", transparent=True)
plt.show()

print(f"Theoretical energy: {E_true:.6f}")
print(f"Predicted energy: {E_pred.item():.6f}")
print(f"Relative error: {100*abs(E_pred.item()-E_true)/E_true:.6f}%")

# %%
