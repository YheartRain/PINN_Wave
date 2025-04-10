import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# PINN for the 1D Wave Equation
#
# PDE:
#   u_tt = c^2 * u_xx
# Domain:
#   t in [0, T], x in [0, L]
# Initial Conditions:
#   u(0, x)  = f1(x) = sin(x)
#   ut(0, x) = f2(x) = 0
# Boundary Conditions:
#   u(t, 0) = 0
#   u(t, L) = 0
#
# This example demonstrates how to set up and train a PINN to solve the 1D wave
# equation. The code is self-contained and uses synthetic data for demonstration. 
################################################################################

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Device configuration (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# Hyperparameters / Problem Setup
################################################################################
L = 1.0      # Spatial domain: x in [0, L]
T = 1.0      # Temporal domain: t in [0, T]
c = 1.0      # Wave speed (can be made learnable if desired)

n_collocation = 2000  # Number of collocation points inside domain
n_boundary    = 100   # Number of points for boundary conditions (each boundary)
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
        
        # Initialize weights (Xavier initialization)
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Forward pass
        for i in range(len(self.linears) - 1):
            x = self.activation(self.linears[i](x))
        # Last layer (output)
        x = self.linears[-1](x)
        return x


################################################################################
# Define the true initial conditions f1(x), f2(x), and boundary conditions
################################################################################
def f1(x):
    # Example initial displacement: a Gaussian or a sine function
    # Here, we use a sine wave for illustration.
    return np.sin(np.pi * x)

def f2(x):
    # Example initial velocity: zero
    return 0.0 * x

def boundary_left(t):
    return 0.0

def boundary_right(t):
    return 0.0


################################################################################
# Exact Solution for the 1D Wave Equation
################################################################################
def exact_solution(x, t, c=1.0):
    """
    Exact analytical solution for the 1D Wave Equation with:
    - Initial condition: u(0,x) = sin(πx)
    - Initial velocity: u_t(0,x) = 0
    - Boundary conditions: u(t,0) = u(t,L) = 0
    
    The solution is: u(x,t) = sin(πx) * cos(cπt)
    """
    return np.sin(np.pi * x) * np.cos(c * np.pi * t)


################################################################################
# Generate Training Data
################################################################################
def generate_training_data():
    # Collocation points in the interior for PDE
    t_coll = np.random.rand(n_collocation, 1) * T
    x_coll = np.random.rand(n_collocation, 1) * L

    # Initial condition points (t=0)
    t_init = np.zeros((n_initial, 1))
    x_init = np.linspace(0, L, n_initial).reshape(-1, 1)

    # Boundary condition points (x=0 and x=L)
    t_bound = np.linspace(0, T, n_boundary).reshape(-1, 1)
    x_bound_left  = np.zeros((n_boundary, 1))
    x_bound_right = L * np.ones((n_boundary, 1))
    
    # f1, f2 values at t=0
    u_init = f1(x_init)
    ut_init = f2(x_init)

    # Boundary values
    u_left  = boundary_left(t_bound)
    u_right = boundary_right(t_bound)

    # Convert all to tensors
    t_coll = torch.tensor(t_coll, dtype=torch.float32, device=device, requires_grad=True)
    x_coll = torch.tensor(x_coll, dtype=torch.float32, device=device, requires_grad=True)
    
    t_init = torch.tensor(t_init, dtype=torch.float32, device=device,requires_grad=True)
    x_init = torch.tensor(x_init, dtype=torch.float32, device=device,requires_grad=True)
    u_init = torch.tensor(u_init, dtype=torch.float32, device=device,requires_grad=True)
    ut_init = torch.tensor(ut_init, dtype=torch.float32, device=device,requires_grad=True)

    t_bound = torch.tensor(t_bound, dtype=torch.float32, device=device,requires_grad=True)
    x_bound_left  = torch.tensor(x_bound_left,  dtype=torch.float32, device=device,requires_grad=True)
    x_bound_right = torch.tensor(x_bound_right, dtype=torch.float32, device=device,requires_grad=True)
    u_left  = torch.tensor(u_left,  dtype=torch.float32, device=device,requires_grad=True)
    u_right = torch.tensor(u_right, dtype=torch.float32, device=device,requires_grad=True)

    return (t_coll, x_coll), (t_init, x_init, u_init, ut_init), \
           (t_bound, x_bound_left, x_bound_right, u_left, u_right)


################################################################################
# Physics-Informed Loss Function
################################################################################
def wave_pde_loss(model, t, x):
    """
    Compute PDE residual for u_tt - c^2 * u_xx = 0.
    """
    # Independent variables require gradients for automatic differentiation
    t.requires_grad_(True)
    x.requires_grad_(True)

    # NN output
    u = model(torch.cat((t, x), dim=1))

    # First derivatives
    grad_u = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = grad_u

    grad_u_t = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    u_tt = grad_u_t

    grad_u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    grad_u_xx = torch.autograd.grad(grad_u_x, x, grad_outputs=torch.ones_like(grad_u_x), create_graph=True)[0]
    u_xx = grad_u_xx

    # PDE residual: u_tt - c^2 * u_xx = 0 -> residual = u_tt - c^2 * u_xx
    residual = u_tt - (c**2) * u_xx

    return torch.mean(residual**2)


def loss_function(model, 
                  t_coll, x_coll, 
                  t_init, x_init, u_init, ut_init,
                  t_bound, x_bound_left, x_bound_right, u_left, u_right):
    # PDE residual on collocation points
    pde_res = wave_pde_loss(model, t_coll, x_coll)

    # Initial condition: u(0, x) = f1(x)
    # Evaluate model at (t=0, x=x_init)
    pred_init = model(torch.cat((t_init, x_init), dim=1))
    ic_loss = torch.mean((pred_init - u_init)**2)

    # Initial velocity: u_t(0, x) = f2(x)
    # We differentiate w.r.t t for t=0
    pred_u_init = pred_init
    grad_init = torch.autograd.grad(pred_u_init, t_init, 
                                    grad_outputs=torch.ones_like(pred_u_init),
                                    create_graph=True)[0]
    ut_loss = torch.mean((grad_init - ut_init)**2)

    # Boundary condition: u(t, 0) = 0
    pred_left = model(torch.cat((t_bound, x_bound_left), dim=1))
    left_bc_loss = torch.mean((pred_left - u_left)**2)

    # Boundary condition: u(t, L) = 0
    pred_right = model(torch.cat((t_bound, x_bound_right), dim=1))
    right_bc_loss = torch.mean((pred_right - u_right)**2)

    # Total loss
    total_loss = pde_res + ic_loss + ut_loss + left_bc_loss + right_bc_loss
    return total_loss


################################################################################
# Plot comparison between exact solution and PINN prediction
################################################################################
def plot_solution_comparison(model, times=[0.59, 0.79, 0.98], save_path=None):
    """
    Create comparison plots between exact solution and PINN prediction 
    at specific time points, similar to Raissi's paper
    """
    fig, axes = plt.subplots(1, len(times), figsize=(15, 4))
    
    # Define x range for plotting
    x_plot = np.linspace(0, L, 200)
    
    for i, t_val in enumerate(times):
        # Create inputs for the model
        t_tensor = torch.ones(x_plot.shape[0], 1, dtype=torch.float32, device=device) * t_val
        x_tensor = torch.tensor(x_plot.reshape(-1, 1), dtype=torch.float32, device=device)
        
        # Get PINN prediction
        with torch.no_grad():
            u_pred = model(torch.cat([t_tensor, x_tensor], dim=1)).cpu().numpy()
        
        # Get exact solution
        u_exact = exact_solution(x_plot, t_val)
        
        # Plot
        axes[i].plot(x_plot, np.abs(u_exact), 'b-', linewidth=2, label='Exact')
        axes[i].plot(x_plot, np.abs(u_pred), 'r--', linewidth=2, label='Prediction')
        axes[i].set_title(f't = {t_val}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('|h(t,x)|')
        axes[i].set_ylim(0, 1.2)  # Adjust as needed based on solution values
        axes[i].grid(True, linestyle='--', alpha=0.6)
        
    # Add legend to the first subplot
    axes[0].legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


################################################################################
# Main Training Loop
################################################################################
def main():
    # Prepare training data
    (t_coll, x_coll), (t_init, x_init, u_init, ut_init), \
    (t_bound, x_bound_left, x_bound_right, u_left, u_right) = generate_training_data()
    
    # Define the PINN model
    layers = [2] + [neurons]*hidden_layers + [1]  # 2 inputs: (t, x), 1 output: u
    model = PINN(layers, activation=nn.Tanh()).to(device)

    # Optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Compute total loss
        loss = loss_function(model, 
                             t_coll, x_coll,
                             t_init, x_init, u_init, ut_init,
                             t_bound, x_bound_left, x_bound_right, u_left, u_right)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch+1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6e}")
    
    # Training complete
    print("Training finished!")
    
    # 修改解决方案：首先生成并保存对比图，然后再显示等高线图
    print("Generating comparison plots first...")
    comparison_times = [0.59, 0.79, 0.98]
    plot_solution_comparison(model, comparison_times, save_path="wave_comparison.png")
    print("Comparison plots saved as 'wave_comparison.png'")
    
    # 等高线图数据准备
    t_test = np.linspace(0, T, 200)
    x_test = np.linspace(0, L, 200)
    T_grid, X_grid = np.meshgrid(t_test, x_test)  # shapes: [200, 200]
    
    # Convert to tensors
    t_test_tensor = torch.tensor(T_grid.flatten()[:, None], dtype=torch.float32, device=device)
    x_test_tensor = torch.tensor(X_grid.flatten()[:, None], dtype=torch.float32, device=device)
    
    # Predict
    with torch.no_grad():
        U_pred = model(torch.cat((t_test_tensor, x_test_tensor), dim=1))
    U_pred = U_pred.cpu().numpy().reshape(200, 200)
    
    # 绘制等高线图
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(T_grid, X_grid, U_pred, 50, cmap="jet")
    plt.colorbar(cp)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("PINN Solution for 1D Wave Equation")
    plt.savefig("wave_contour.png", dpi=300, bbox_inches='tight')
    print("Contour plot saved as 'wave_contour.png'")
    
    # 显示对比图
    comparison_fig = plt.figure(figsize=(15, 4))
    img = plt.imread("wave_comparison.png")
    plt.imshow(img)
    plt.axis('off')
    
    # 显示所有图
    plt.show()
    
    print("\nAll figures have been saved. Please check your directory for 'wave_comparison.png' and 'wave_contour.png'")

if __name__ == "__main__":
    main()