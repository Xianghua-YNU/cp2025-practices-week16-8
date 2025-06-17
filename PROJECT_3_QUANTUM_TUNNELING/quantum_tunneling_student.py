import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class QuantumTunnelingSolver:
    def __init__(self, Nx=220, Nt=300, x0=40, k0=0.5, d=10, barrier_width=3, barrier_height=1.0):
        self.Nx = Nx
        self.Nt = Nt
        self.x0 = x0
        self.k0 = k0
        self.d = d
        self.barrier_width = barrier_width
        self.barrier_height = barrier_height
        self.x = np.arange(self.Nx)
        self.V = self.setup_potential()
        self.C = np.zeros((self.Nx, self.Nt), complex)
        self.B = np.zeros((self.Nx, self.Nt), complex)

    def wavefun(self, x):
        """Gaussian wave packet function"""
        return np.exp(self.k0*1j*x)*np.exp(-(x-self.x0)**2*np.log10(2)/self.d**2)

    def setup_potential(self):
        """Setup potential barrier"""
        self.V = np.zeros(self.Nx)
        self.V[self.Nx//2:self.Nx//2+self.barrier_width] = self.barrier_height
        return self.V

    def build_coefficient_matrix(self):
        """Build coefficient matrix for Crank-Nicolson scheme"""
        A = np.diag(-2+2j-self.V) + np.diag(np.ones(self.Nx-1),1) + np.diag(np.ones(self.Nx-1),-1)
        return A

    def solve_schrodinger(self):
        """Solve 1D time-dependent Schrodinger equation using Crank-Nicolson method"""
        A = self.build_coefficient_matrix()
        
        self.B[:,0] = self.wavefun(self.x)
        
        for t in range(self.Nt-1):
            self.C[:,t+1] = 4j*np.linalg.solve(A, self.B[:,t])
            self.B[:,t+1] = self.C[:,t+1] - self.B[:,t]
        
        return self.x, self.V, self.B, self.C

    def calculate_coefficients(self):
        """Calculate transmission and reflection coefficients"""
        barrier_position = len(self.x)//2
        transmitted_prob = np.sum(np.abs(self.B[barrier_position+self.barrier_width:, -1])**2)
        reflected_prob = np.sum(np.abs(self.B[:barrier_position, -1])**2)
        total_prob = np.sum(np.abs(self.B[:, -1])**2)
        return transmitted_prob/total_prob, reflected_prob/total_prob

    def plot_evolution(self, time_indices=None):
        """Plot wave function evolution at specific times"""
        if time_indices is None:
            Nt = self.B.shape[1]
            time_indices = [0, Nt//4, Nt//2, 3*Nt//4, Nt-1]
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        # Add overall title with barrier parameters
        fig.suptitle(f'Quantum Tunneling Evolution - Barrier Width: {self.barrier_width}, Barrier Height: {self.barrier_height}', 
                     fontsize=14, fontweight='bold')
        
        for i, t_idx in enumerate(time_indices):
            if i < len(axes):
                ax = axes[i]
                
                # Plot probability density
                prob_density = np.abs(self.B[:, t_idx])**2
                ax.plot(self.x, prob_density, 'b-', linewidth=2, 
                       label=f'|ψ|² at t={t_idx}')
                
                # Plot potential
                ax.plot(self.x, self.V, 'k-', linewidth=2, 
                       label=f'Barrier (Width={self.barrier_width}, Height={self.barrier_height})')
                
                ax.set_xlabel('Position')
                ax.set_ylabel('Probability Density')
                ax.set_title(f'Time step: {t_idx}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(len(time_indices), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()

    def create_animation(self, interval=20):
        """Create animation of wave packet evolution"""
        Nx, Nt = self.B.shape
        
        fig = plt.figure(figsize=(10, 6))
        plt.axis([0, Nx, 0, np.max(self.V)*1.1])
        
        # Add title with barrier parameters
        plt.title(f'Quantum Tunneling Animation - Barrier Width: {self.barrier_width}, Barrier Height: {self.barrier_height}', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Position')
        plt.ylabel('Probability Density / Potential')
        
        myline, = plt.plot([], [], 'r', lw=2, label='|ψ|²')
        myline1, = plt.plot(self.x, self.V, 'k', lw=2, 
                           label=f'Barrier (Width={self.barrier_width}, Height={self.barrier_height})')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        def animate(i):
            myline.set_data(self.x, np.abs(self.B[:, i])**2)
            myline1.set_data(self.x, self.V)
            return myline, myline1
        
        anim = animation.FuncAnimation(fig, animate, frames=Nt, interval=interval)
        return anim

    def verify_probability_conservation(self):
        """Verify probability conservation during evolution"""
        total_prob = np.zeros(self.Nt)
        for t in range(self.Nt):
            total_prob[t] = np.sum(np.abs(self.B[:, t])**2)
        
        return total_prob

    def demonstrate(self):
        """Demonstration function for quantum tunneling"""
        print("Quantum Tunneling Simulation")
        print("=" * 40)
        
        # Solve the equation
        print("Solving Schrodinger equation...")
        self.solve_schrodinger()
        T, R = self.calculate_coefficients()
        
        print(f"\n势垒宽度:{self.barrier_width}, 势垒高度:{self.barrier_height} 结果")
        print(f"Transmission coefficient: {T:.4f}")
        print(f"Reflection coefficient: {R:.4f}")
        print(f"Total (T + R): {T + R:.4f}")
        
        # Plot evolution
        print("\nPlotting wave function evolution...")
        self.plot_evolution()
        
        # Check probability conservation
        total_prob = self.verify_probability_conservation()
        print(f"\nProbability conservation:")
        print(f"Initial probability: {total_prob[0]:.6f}")
        print(f"Final probability: {total_prob[-1]:.6f}")
        print(f"Relative change: {abs(total_prob[-1] - total_prob[0])/total_prob[0]*100:.4f}%")
        
        # Create animation
        print("\nCreating animation...")
        anim = self.create_animation()
        plt.show()
        
        return anim


def demonstrate_quantum_tunneling():
    """便捷的演示函数"""
    solver = QuantumTunnelingSolver()
    return solver.demonstrate()


if __name__ == "__main__":
    # 运行演示
    barrier_width = 3
    barrier_height = 1.0
    solver = QuantumTunnelingSolver(barrier_width=barrier_width, barrier_height=barrier_height)
    animation = solver.demonstrate()
