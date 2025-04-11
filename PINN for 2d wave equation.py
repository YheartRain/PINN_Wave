import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

################################################################################
# PINN for the 2D Wave Equation
#
# PDE:
#   u_tt = c^2 * (u_xx + u_yy)
# Domain:
#   t in [0, T], x in [0, Lx], y in [0, Ly]
# Initial Conditions:
#   u(0, x, y)  = f1(x, y) = sin(pi*x) * sin(pi*y)
#   ut(0, x, y) = f2(x, y) = 0
# Boundary Conditions:
#   u(t, 0, y) = u(t, Lx, y) = u(t, x, 0) = u(t, x, Ly) = 0
#
################################################################################

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# Hyperparameters / Problem Setup
################################################################################
Lx = 1.0     # Spatial domain: x in [0, Lx]
Ly = 1.0     # Spatial domain: y in [0, Ly]
T = 1.0      # Temporal domain: t in [0, T]
c = 1.0      # Wave speed

n_collocation = 5000  # Number of collocation points inside domain
n_boundary    = 500   # Number of points for boundary conditions (each boundary)
n_initial     = 1000  # Number of points for initial conditions
hidden_layers = 6     # Number of hidden layers
neurons       = 50    # Neurons per hidden layer
learning_rate = 1e-3  # Learning rate for optimizer
num_epochs    = 10000 # Training epochs

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
        
        # Initialize weights (Xavier initialization)
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Forward pass
        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
        # output
        x = self.linears[-1](x)
        return x


################################################################################
# Define the true initial conditions f1(x,y), f2(x,y), and boundary conditions
################################################################################
def f1(x, y):
    # initial displacement: sin(pi*x) * sin(pi*y)
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f2(x, y):
    # initial velocity: zero
    return np.zeros_like(x)

def boundary_condition(t, x, y):
    # Dirichlet boundary conditions (all edges)
    return np.zeros_like(t)


################################################################################
# Exact Solution for the 2D Wave Equation (if available)
################################################################################
def exact_solution(x, y, t, c=1.0):
    """
    Exact analytical solution for the 2D Wave Equation with:
    - Initial condition: u(0,x,y) = sin(πx) * sin(πy)
    - Initial velocity: u_t(0,x,y) = 0
    - Boundary conditions: u(t,0,y) = u(t,Lx,y) = u(t,x,0) = u(t,x,Ly) = 0
    
    The solution is: u(x,y,t) = sin(πx) * sin(πy) * cos(c*sqrt(2)*πt)
    """
    omega = np.pi * c * np.sqrt(2)  # Frequency for the (1,1) mode
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(omega * t)


################################################################################
# Generate Training Data
################################################################################
def generate_training_data():
    # Collocation points in the interior for PDE
    t_coll = np.random.rand(n_collocation, 1) * T
    x_coll = np.random.rand(n_collocation, 1) * Lx
    y_coll = np.random.rand(n_collocation, 1) * Ly

    # Initial condition points (t=0)
    grid_size = int(np.sqrt(n_initial))
    x_init_grid, y_init_grid = np.meshgrid(np.linspace(0, Lx, grid_size), 
                                          np.linspace(0, Ly, grid_size))
    x_init = x_init_grid.flatten().reshape(-1, 1)
    y_init = y_init_grid.flatten().reshape(-1, 1)
    t_init = np.zeros_like(x_init)  # t=0 for all initial points

    # Boundary points (Sample points for each boundary)
    # Bottom boundary (y=0)
    t_bound_bottom = np.random.rand(n_boundary, 1) * T
    x_bound_bottom = np.random.rand(n_boundary, 1) * Lx
    y_bound_bottom = np.zeros((n_boundary, 1))
    
    # Top boundary (y=Ly)
    t_bound_top = np.random.rand(n_boundary, 1) * T
    x_bound_top = np.random.rand(n_boundary, 1) * Lx
    y_bound_top = Ly * np.ones((n_boundary, 1))
    
    # Left boundary (x=0)
    t_bound_left = np.random.rand(n_boundary, 1) * T
    x_bound_left = np.zeros((n_boundary, 1))
    y_bound_left = np.random.rand(n_boundary, 1) * Ly
    
    # Right boundary (x=Lx)
    t_bound_right = np.random.rand(n_boundary, 1) * T
    x_bound_right = Lx * np.ones((n_boundary, 1))
    y_bound_right = np.random.rand(n_boundary, 1) * Ly
    
    # Values for initial conditions
    u_init = f1(x_init, y_init)
    ut_init = f2(x_init, y_init)

    # Values for boundary conditions (all zeros for this example)
    u_bound_bottom = boundary_condition(t_bound_bottom, x_bound_bottom, y_bound_bottom)
    u_bound_top = boundary_condition(t_bound_top, x_bound_top, y_bound_top)
    u_bound_left = boundary_condition(t_bound_left, x_bound_left, y_bound_left)
    u_bound_right = boundary_condition(t_bound_right, x_bound_right, y_bound_right)

    # Convert everything to tensors
    t_coll = torch.tensor(t_coll, dtype=torch.float32, device=device, requires_grad=True)
    x_coll = torch.tensor(x_coll, dtype=torch.float32, device=device, requires_grad=True)
    y_coll = torch.tensor(y_coll, dtype=torch.float32, device=device, requires_grad=True)
    
    t_init = torch.tensor(t_init, dtype=torch.float32, device=device, requires_grad=True)
    x_init = torch.tensor(x_init, dtype=torch.float32, device=device, requires_grad=True)
    y_init = torch.tensor(y_init, dtype=torch.float32, device=device, requires_grad=True)
    u_init = torch.tensor(u_init, dtype=torch.float32, device=device)
    ut_init = torch.tensor(ut_init, dtype=torch.float32, device=device)
    
    # Tensors for boundary points and values
    # Bottom boundary
    t_bound_bottom = torch.tensor(t_bound_bottom, dtype=torch.float32, device=device, requires_grad=True)
    x_bound_bottom = torch.tensor(x_bound_bottom, dtype=torch.float32, device=device, requires_grad=True)
    y_bound_bottom = torch.tensor(y_bound_bottom, dtype=torch.float32, device=device, requires_grad=True)
    u_bound_bottom = torch.tensor(u_bound_bottom, dtype=torch.float32, device=device)
    
    # Top boundary
    t_bound_top = torch.tensor(t_bound_top, dtype=torch.float32, device=device, requires_grad=True)
    x_bound_top = torch.tensor(x_bound_top, dtype=torch.float32, device=device, requires_grad=True)
    y_bound_top = torch.tensor(y_bound_top, dtype=torch.float32, device=device, requires_grad=True)
    u_bound_top = torch.tensor(u_bound_top, dtype=torch.float32, device=device)
    
    # Left boundary
    t_bound_left = torch.tensor(t_bound_left, dtype=torch.float32, device=device, requires_grad=True)
    x_bound_left = torch.tensor(x_bound_left, dtype=torch.float32, device=device, requires_grad=True)
    y_bound_left = torch.tensor(y_bound_left, dtype=torch.float32, device=device, requires_grad=True)
    u_bound_left = torch.tensor(u_bound_left, dtype=torch.float32, device=device)
    
    # Right boundary
    t_bound_right = torch.tensor(t_bound_right, dtype=torch.float32, device=device, requires_grad=True)
    x_bound_right = torch.tensor(x_bound_right, dtype=torch.float32, device=device, requires_grad=True)
    y_bound_right = torch.tensor(y_bound_right, dtype=torch.float32, device=device, requires_grad=True)
    u_bound_right = torch.tensor(u_bound_right, dtype=torch.float32, device=device)

    # Bundle data
    collocation_pts = (t_coll, x_coll, y_coll)
    initial_pts = (t_init, x_init, y_init, u_init, ut_init)
    boundary_pts = {
        'bottom': (t_bound_bottom, x_bound_bottom, y_bound_bottom, u_bound_bottom),
        'top': (t_bound_top, x_bound_top, y_bound_top, u_bound_top),
        'left': (t_bound_left, x_bound_left, y_bound_left, u_bound_left),
        'right': (t_bound_right, x_bound_right, y_bound_right, u_bound_right)
    }

    return collocation_pts, initial_pts, boundary_pts


################################################################################
# Physics-Informed Loss Function
################################################################################
def wave_pde_loss(model, t, x, y):
    """
    Compute PDE residual for u_tt = c^2 * (u_xx + u_yy).
    """
    # Ensure gradients are computed
    t.requires_grad_(True)
    x.requires_grad_(True)
    y.requires_grad_(True)

    # NN output
    inputs = torch.cat((t, x, y), dim=1)
    u = model(inputs)

    # First time derivative
    u_t = torch.autograd.grad(u, t, 
                             grad_outputs=torch.ones_like(u), 
                             create_graph=True)[0]

    # Second time derivative
    u_tt = torch.autograd.grad(u_t, t, 
                              grad_outputs=torch.ones_like(u_t),
                              create_graph=True)[0]

    # First x derivative
    u_x = torch.autograd.grad(u, x, 
                             grad_outputs=torch.ones_like(u),
                             create_graph=True)[0]

    # Second x derivative
    u_xx = torch.autograd.grad(u_x, x,
                              grad_outputs=torch.ones_like(u_x),
                              create_graph=True)[0]
    
    # First y derivative
    u_y = torch.autograd.grad(u, y, 
                             grad_outputs=torch.ones_like(u),
                             create_graph=True)[0]

    # Second y derivative
    u_yy = torch.autograd.grad(u_y, y,
                              grad_outputs=torch.ones_like(u_y),
                              create_graph=True)[0]

    # PDE residual: u_tt - c^2 * (u_xx + u_yy) = 0
    residual = u_tt - (c**2) * (u_xx + u_yy)

    return torch.mean(residual**2)


def initial_condition_loss(model, t, x, y, u_true, ut_true):
    """
    Compute loss for initial conditions: u(0,x,y) = f1(x,y) and u_t(0,x,y) = f2(x,y)
    """
    # Initial displacement: u(0,x,y) = f1(x,y)
    inputs = torch.cat((t, x, y), dim=1)
    u_pred = model(inputs)
    ic_loss = torch.mean((u_pred - u_true)**2)
    
    # Initial velocity: u_t(0,x,y) = f2(x,y)
    # Compute u_t by differentiating with respect to t
    u_t = torch.autograd.grad(u_pred, t, 
                              grad_outputs=torch.ones_like(u_pred),
                              create_graph=True)[0]
    
    ut_loss = torch.mean((u_t - ut_true)**2)
    
    return ic_loss + ut_loss


def boundary_condition_loss(model, boundary_pts):
    """
    Compute loss for boundary conditions at all four edges of the domain.
    """
    bc_loss = 0
    
    # Process each boundary
    for boundary_name, (t, x, y, u_true) in boundary_pts.items():
        inputs = torch.cat((t, x, y), dim=1)
        u_pred = model(inputs)
        bc_loss += torch.mean((u_pred - u_true)**2)
    
    return bc_loss


def loss_function(model, collocation_pts, initial_pts, boundary_pts):
    """
    Compute total loss for the PINN.
    """
    # Unpack data
    t_coll, x_coll, y_coll = collocation_pts
    t_init, x_init, y_init, u_init, ut_init = initial_pts
    
    # PDE residual loss
    pde_loss = wave_pde_loss(model, t_coll, x_coll, y_coll)
    
    # Initial condition loss
    ic_loss = initial_condition_loss(model, t_init, x_init, y_init, u_init, ut_init)
    
    # Boundary condition loss
    bc_loss = boundary_condition_loss(model, boundary_pts)
    
    # Combine losses with appropriate weights
    total_loss = pde_loss + ic_loss + bc_loss
    
    return total_loss, pde_loss, ic_loss, bc_loss


################################################################################
# Visualization Functions
################################################################################
def visualize_solution(model, time_slices=[0.25, 0.5, 0.75, 1.0], save_path=None):
    """
    Plot the PINN solution at several time points and compare with exact solution.
    """
    fig = plt.figure(figsize=(16, 12))
    x_grid = np.linspace(0, Lx, 50)
    y_grid = np.linspace(0, Ly, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    for i, t_val in enumerate(time_slices):
        # Prepare input tensors 
        t_tensor = torch.full((X.size,1), t_val, dtype=torch.float32, device=device)
        x_tensor = torch.tensor(X.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)
        y_tensor = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).reshape(-1, 1)
        
        # Predict using PINN
        with torch.no_grad():
            inputs = torch.cat((t_tensor, x_tensor, y_tensor), dim=1)
            u_pred = model(inputs).cpu().numpy().reshape(X.shape)
        
        # Compute exact solution
        u_exact = exact_solution(X, Y, t_val)
        
        # Plot PINN solution
        ax1 = fig.add_subplot(len(time_slices), 3, 3*i+1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, u_pred, cmap=cm.coolwarm)
        ax1.set_title(f'PINN Solution at t={t_val}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        
        # Plot exact solution
        ax2 = fig.add_subplot(len(time_slices), 3, 3*i+2, projection='3d')
        surf2 = ax2.plot_surface(X, Y, u_exact, cmap=cm.coolwarm)
        ax2.set_title(f'Exact Solution at t={t_val}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        
        # Plot absolute error
        ax3 = fig.add_subplot(len(time_slices), 3, 3*i+3, projection='3d')
        error = np.abs(u_pred - u_exact)
        surf3 = ax3.plot_surface(X, Y, error, cmap=cm.Reds)
        ax3.set_title(f'Absolute Error at t={t_val}')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('|error|')
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


################################################################################
# Main Training Loop
################################################################################
def main():
    # Generate training data
    collocation_pts, initial_pts, boundary_pts = generate_training_data()
    
    # Create model
    layers = [3] + [neurons]*hidden_layers + [1]  # 3 inputs: (t, x, y), 1 output: u
    model = PINN(layers, activation=nn.Tanh()).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler - removed verbose parameter to avoid warning
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1000)
    
    # Training history
    loss_history = []
    pde_loss_history = []
    ic_loss_history = []
    bc_loss_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Compute loss
        loss, pde_loss, ic_loss, bc_loss = loss_function(model, collocation_pts, initial_pts, boundary_pts)
        
        # Store loss values
        loss_history.append(loss.item())
        pde_loss_history.append(pde_loss.item())
        ic_loss_history.append(ic_loss.item())
        bc_loss_history.append(bc_loss.item())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step(loss)
        
        # Print progress
        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Loss: {loss.item():.6e}, "
                  f"PDE Loss: {pde_loss.item():.6e}, "
                  f"IC Loss: {ic_loss.item():.6e}, "
                  f"BC Loss: {bc_loss.item():.6e}")
    

    print("Training completed!")
    
    # Visualize solution
    visualize_solution(model, save_path="2d_wave_solution.png")
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history, label='Total Loss')
    plt.semilogy(pde_loss_history, label='PDE Loss')
    plt.semilogy(ic_loss_history, label='IC Loss')
    plt.semilogy(bc_loss_history, label='BC Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.savefig("2d_wave_loss_history.png", dpi=300, bbox_inches='tight')
    
    # Show figures
    plt.show()


if __name__ == "__main__":
    main()