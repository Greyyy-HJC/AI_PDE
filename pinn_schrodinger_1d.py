# %%
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np


# ===== Parameters =====
L = 1.0   # Well width
n = 1     # Quantum number
hbar = 1.0
m = 1.0
# Theoretical energy eigenvalue
E_true = (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)

# ===== Generate training data =====
x = np.linspace(0, L, 200)[:, None]          # Spatial grid (200,1) - increased resolution
t = np.linspace(0, 1, 100)[:, None]          # Time grid (100,1) - increased resolution
Xg, Tg = np.meshgrid(x, t)
# Input (N,2)
X_train = np.concatenate([Xg.reshape(-1,1), Tg.reshape(-1,1)], axis=1)
# Analytical solution (real and imaginary parts)
psi_real = np.sqrt(2/L) * np.sin(n * np.pi * Xg.reshape(-1,1) / L) * np.cos(E_true * Tg.reshape(-1,1) / hbar)
psi_imag = -np.sqrt(2/L) * np.sin(n * np.pi * Xg.reshape(-1,1) / L) * np.sin(E_true * Tg.reshape(-1,1) / hbar)

# Convert to torch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
Y_train = torch.cat([torch.tensor(psi_real, dtype=torch.float32, device=device),
                     torch.tensor(psi_imag, dtype=torch.float32, device=device)], dim=1)

# ===== Neural Network Definition =====
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Improved network architecture with more layers and units
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
        
        # Initialize weights for better convergence
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, xt):
        return self.net(xt)

net = PINN().to(device)
# Learnable energy parameter
E_pred = nn.Parameter(torch.tensor(E_true * 0.9, device=device))  # Start with a worse initial guess

# ===== Training Configuration =====
num_epochs = 3000  # Increased number of epochs
lr = 5e-4  # Lower learning rate for stability
torch.manual_seed(0)
optimizer = torch.optim.Adam(list(net.parameters()) + [E_pred], lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5, verbose=True)

# ===== Loss Function =====
def compute_loss(model, X, Y):
    # Create input with gradient tracking
    X_with_grad = X.clone().detach().requires_grad_(True)
    # Forward pass
    uv = model(X_with_grad)
    u = uv[:, 0:1]  # Real part
    v = uv[:, 1:2]  # Imaginary part
    # Data loss
    data_loss = torch.mean((u - Y[:,0:1])**2 + (v - Y[:,1:2])**2)
    
    # Compute derivatives (with respect to X components)
    # Second-order spatial derivatives
    du_dx = autograd.grad(u.sum(), X_with_grad, create_graph=True)[0][:, 0:1]
    d2u_dx2 = autograd.grad(du_dx.sum(), X_with_grad, create_graph=True)[0][:, 0:1]
    dv_dx = autograd.grad(v.sum(), X_with_grad, create_graph=True)[0][:, 0:1]
    d2v_dx2 = autograd.grad(dv_dx.sum(), X_with_grad, create_graph=True)[0][:, 0:1]
    
    # Time derivatives
    du_dt = autograd.grad(u.sum(), X_with_grad, create_graph=True)[0][:, 1:2]
    dv_dt = autograd.grad(v.sum(), X_with_grad, create_graph=True)[0][:, 1:2]
    
    # Stationary residual
    res_sr = (-hbar**2/(2*m)) * d2u_dx2 - E_pred * u
    res_si = (-hbar**2/(2*m)) * d2v_dx2 - E_pred * v
    
    # Time-dependent residual
    res_tr = hbar * du_dt - (hbar**2/(2*m)) * d2v_dx2
    res_ti = hbar * dv_dt + (hbar**2/(2*m)) * d2u_dx2
    
    # Physics loss (weighted higher to enforce physical constraints)
    physics_loss = 10.0 * torch.mean(res_sr**2 + res_si**2 + res_tr**2 + res_ti**2)
    total_loss = data_loss + physics_loss
    return total_loss, data_loss, physics_loss

# %%
# ===== Training Loop =====
best_loss = float('inf')
for epoch in range(1, num_epochs+1):
    optimizer.zero_grad()
    loss, d_loss, p_loss = compute_loss(net, X_train, Y_train)
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    
    optimizer.step()
    scheduler.step(loss)
    
    if loss < best_loss:
        best_loss = loss
        best_model_state = {
            'model': net.state_dict(),
            'E_pred': E_pred.item()
        }
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4e}, DataLoss: {d_loss.item():.4e}, PhysicsLoss: {p_loss.item():.4e}, E_pred: {E_pred.item():.4f}")

# Load the best model
net.load_state_dict(best_model_state['model'])
E_pred.data = torch.tensor(best_model_state['E_pred'], device=device)

# %%
# ===== Results Visualization =====
import matplotlib.pyplot as plt
x_test = np.linspace(0, L, 200)[:,None]
t_test = np.zeros_like(x_test)
X_test = torch.tensor(np.concatenate([x_test, t_test], axis=1), dtype=torch.float32, device=device)
with torch.no_grad():
    psi_pred = net(X_test)
psi_real_pred = psi_pred[:,0].cpu().numpy()
psi_theory = np.sqrt(2/L) * np.sin(n * np.pi * x_test[:,0] / L)

plt.figure(figsize=(10, 6))
plt.plot(x_test[:,0], psi_real_pred, 'r-', linewidth=2, label='PINN')
plt.plot(x_test[:,0], psi_theory, 'b--', linewidth=1.5, label='Theory')
plt.xlabel('x', fontsize=14)
plt.ylabel('$\psi(x)$', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.title('Wavefunction Comparison (t=0)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pinn_schrodinger_1d.pdf", transparent=True)
plt.show()

print(f"Theoretical energy: {E_true:.4f}")
print(f"Predicted energy: {E_pred.item():.4f}")
print(f"Relative error: {100*abs(E_pred.item()-E_true)/E_true:.4f}%")



# %%
