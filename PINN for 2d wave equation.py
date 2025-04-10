import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# PINN for the 1D Wave Equation
################################################################################

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# Hyperparameters / Problem Setup
################################################################################
L = 1.0      # Spatial domain: x in [0, L]
T = 1.0      # Temporal domain: t in [0, T]
c = 1.0      # Wave speed

n_collocation = 2000  # Number of collocation points inside domain
n_boundary    = 100   # Number of points for boundary conditions
n_initial     = 100   # Number of points for initial conditions
hidden_layers = 5     # Number of hidden layers
neurons       = 50    # Neurons per hidden layer
learning_rate = 1e-3  # Learning rate for optimizer
num_epochs    = 5000  # Training epochs

################################################################################
# Neural Network Definition
################################################################################
class PINN(nn.Module):
    def __init__(self, layers, activation=nn.Tanh()):
        super(PINN, self).__init__()
        
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        
        # Initialize weights
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
        x = self.linears[-1](x)
        return x

################################################################################
# Define initial and boundary conditions
################################################################################
def f1(x):
    return np.sin(np.pi * x)

def f2(x):
    return 0.0 * x

def boundary_left(t):
    return 0.0

def boundary_right(t):
    return 0.0

################################################################################
# Exact Solution
################################################################################
def exact_solution(x, t, c=1.0):
    return np.sin(np.pi * x) * np.cos(c * np.pi * t)

################################################################################
# Generate Training Data
################################################################################
def generate_training_data():
    # Collocation points
    t_coll = np.random.rand(n_collocation, 1) * T
    x_coll = np.random.rand(n_collocation, 1) * L

    # Initial condition points
    t_init = np.zeros((n_initial, 1))
    x_init = np.linspace(0, L, n_initial).reshape(-1, 1)

    # Boundary points
    t_bound = np.linspace(0, T, n_boundary).reshape(-1, 1)
    x_bound_left = np.zeros((n_boundary, 1))
    x_bound_right = L * np.ones((n_boundary, 1))
    
    # Initial values
    u_init = f1(x_init)
    ut_init = f2(x_init)

    # Boundary values
    u_left = boundary_left(t_bound)
    u_right = boundary_right(t_bound)

    # Convert to tensors
    t_coll = torch.tensor(t_coll, dtype=torch.float32, device=device, requires_grad=True)
    x_coll = torch.tensor(x_coll, dtype=torch.float32, device=device, requires_grad=True)
    
    t_init = torch.tensor(t_init, dtype=torch.float32, device=device, requires_grad=True)
    x_init = torch.tensor(x_init, dtype=torch.float32, device=device, requires_grad=True)
    u_init = torch.tensor(u_init, dtype=torch.float32, device=device)
    ut_init = torch.tensor(ut_init, dtype=torch.float32, device=device)

    t_bound = torch.tensor(t_bound, dtype=torch.float32, device=device, requires_grad=True)
    x_bound_left = torch.tensor(x_bound_left, dtype=torch.float32, device=device, requires_grad=True)
    x_bound_right = torch.tensor(x_bound_right, dtype=torch.float32, device=device, requires_grad=True)
    u_left = torch.tensor(u_left, dtype=torch.float32, device=device)
    u_right = torch.tensor(u_right, dtype=torch.float32, device=device)

    return (t_coll, x_coll), (t_init, x_init, u_init, ut_init), \
           (t_bound, x_bound_left, x_bound_right, u_left, u_right)

################################################################################
# Physics-Informed Loss Function
################################################################################
def wave_pde_loss(model, t, x):
    t.requires_grad_(True)
    x.requires_grad_(True)

    u = model(torch.cat((t, x), dim=1))

    # First derivatives
    grad_u = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grad_u

    grad_u_t = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    u_tt = grad_u_t

    grad_u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    grad_u_xx = torch.autograd.grad(grad_u_x, x, grad_outputs=torch.ones_like(grad_u_x), create_graph=True)[0]
    u_xx = grad_u_xx

    residual = u_tt - (c**2) * u_xx

    return torch.mean(residual**2)

def loss_function(model, 
                  t_coll, x_coll, 
                  t_init, x_init, u_init, ut_init,
                  t_bound, x_bound_left, x_bound_right, u_left, u_right):
    # PDE loss
    pde_loss = wave_pde_loss(model, t_coll, x_coll)

    # Initial condition loss
    pred_init = model(torch.cat((t_init, x_init), dim=1))
    ic_loss = torch.mean((pred_init - u_init)**2)

    # Initial velocity loss
    pred_u_init = pred_init
    grad_init = torch.autograd.grad(pred_u_init, t_init, 
                                    grad_outputs=torch.ones_like(pred_u_init),
                                    create_graph=True)[0]
    ut_loss = torch.mean((grad_init - ut_init)**2)

    # Boundary losses
    pred_left = model(torch.cat((t_bound, x_bound_left), dim=1))
    pred_right = model(torch.cat((t_bound, x_bound_right), dim=1))
    
    left_bc_loss = torch.mean((pred_left - u_left)**2)
    right_bc_loss = torch.mean((pred_right - u_right)**2)

    # Total loss
    total_loss = pde_loss + ic_loss + ut_loss + left_bc_loss + right_bc_loss
    return total_loss, pde_loss, ic_loss, ut_loss, left_bc_loss, right_bc_loss

################################################################################
# Visualization Functions
################################################################################
def plot_training_points(t_coll, x_coll, t_init, x_init, t_bound, x_bound_left, x_bound_right):
    plt.figure(figsize=(8, 6))
    plt.scatter(t_coll.detach().cpu().numpy(), x_coll.detach().cpu().numpy(), 
               c='blue', alpha=0.5, s=1, label='Collocation points')
    plt.scatter(t_init.detach().cpu().numpy(), x_init.detach().cpu().numpy(), 
               c='red', s=5, label='Initial points')
    plt.scatter(t_bound.detach().cpu().numpy(), x_bound_left.detach().cpu().numpy(), 
               c='green', s=5, label='Left boundary')
    plt.scatter(t_bound.detach().cpu().numpy(), x_bound_right.detach().cpu().numpy(), 
               c='yellow', s=5, label='Right boundary')
    
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Distribution of Training Points')
    plt.legend()
    plt.show()

def plot_solution_comparison(model, times=[0.59, 0.79, 0.98]):
    fig, axes = plt.subplots(1, len(times), figsize=(15, 4))
    x_plot = np.linspace(0, L, 200)
    
    for i, t_val in enumerate(times):
        t_tensor = torch.ones(x_plot.shape[0], 1, dtype=torch.float32, device=device) * t_val
        x_tensor = torch.tensor(x_plot.reshape(-1, 1), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            u_pred = model(torch.cat([t_tensor, x_tensor], dim=1)).cpu().numpy()
        
        u_exact = exact_solution(x_plot, t_val)
        
        axes[i].plot(x_plot, np.abs(u_exact), 'b-', linewidth=2, label='Exact')
        axes[i].plot(x_plot, np.abs(u_pred), 'r--', linewidth=2, label='Prediction')
        axes[i].set_title(f't = {t_val}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('|h(t,x)|')
        axes[i].set_ylim(0, 1.2)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
    axes[0].legend()
    plt.tight_layout()
    plt.show()

################################################################################
# Main Training Loop
################################################################################
def main():
    # Prepare training data
    (t_coll, x_coll), (t_init, x_init, u_init, ut_init), \
    (t_bound, x_bound_left, x_bound_right, u_left, u_right) = generate_training_data()
    
    # Plot training points distribution
    print("Plotting training points distribution...")
    plot_training_points(t_coll, x_coll, t_init, x_init, t_bound, x_bound_left, x_bound_right)
    
    # Define the PINN model
    layers = [2] + [neurons]*hidden_layers + [1]
    model = PINN(layers, activation=nn.Tanh()).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize loss history
    losses = []
    pde_losses = []
    ic_losses = []
    ut_losses = []
    bc_left_losses = []
    bc_right_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Compute losses
        total_loss, pde_loss, ic_loss, ut_loss, left_bc_loss, right_bc_loss = loss_function(
            model, t_coll, x_coll, t_init, x_init, u_init, ut_init,
            t_bound, x_bound_left, x_bound_right, u_left, u_right
        )
        
        # Record losses
        losses.append(total_loss.item())
        pde_losses.append(pde_loss.item())
        ic_losses.append(ic_loss.item())
        ut_losses.append(ut_loss.item())
        bc_left_losses.append(left_bc_loss.item())
        bc_right_losses.append(right_bc_loss.item())
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        if (epoch+1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.6e}")
    
    print("Training finished!")
    
    # Plot solution comparison
    print("Generating comparison plots...")
    plot_solution_comparison(model)
    
    # Plot contour
    plt.figure(figsize=(8, 6))
    t_test = np.linspace(0, T, 200)
    x_test = np.linspace(0, L, 200)
    T_grid, X_grid = np.meshgrid(t_test, x_test)
    
    t_test_tensor = torch.tensor(T_grid.flatten()[:, None], dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(X_grid.flatten()[:, None], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        U_pred = model(torch.cat((t_test_tensor, x_test_tensor), dim=1))
    U_pred = U_pred.cpu().numpy().reshape(200, 200)
    
    cp = plt.contourf(T_grid, X_grid, U_pred, 50, cmap="jet")
    plt.colorbar(cp)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("PINN Solution for 1D Wave Equation")
    plt.show()
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses, label='Total Loss')
    plt.semilogy(pde_losses, label='PDE Loss')
    plt.semilogy(ic_losses, label='IC Loss')
    plt.semilogy(ut_losses, label='Velocity Loss')
    plt.semilogy(bc_left_losses, label='Left BC Loss')
    plt.semilogy(bc_right_losses, label='Right BC Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()