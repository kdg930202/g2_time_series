import numpy as np
from qutip import Qobj, basis, tensor, identity, sigmax, sigmaz, mesolve, qzero, expect



# --- 1. Model Parameters ---
N = 3              # Number of qubits (N-qubit system)
delta_t = 0.1      # Time step for unitary evolution (Δt)
hbar = 1           # Reduced Planck constant (set to 1 for simplicity)

# --- 2. Initial Setup: Operators ---

# Define basic Pauli operators and identity for a single qubit
sigma_x = sigmax()
sigma_z = sigmaz()
I = identity(2)

def get_n_qubit_op(op, k, N):
    """Returns the tensor product for 'op' acting on the k-th qubit of an N-qubit system."""
    op_list = [I] * N
    op_list[k] = op
    return tensor(op_list)

def get_n_qubit_op(op, k, N):
    """Returns the tensor product for 'op' acting on the k-th qubit of an N-qubit system."""
    op_list = [I] * N
    op_list[k] = op
    return tensor(op_list)


def build_hamiltonian(N):
    dims = [[2] * N]
    H = qzero(dims)
    
    # Randomly sample the coupling (J_ij) and transverse field (h_x^(i)) terms
    J_mat = np.random.uniform(-1, 1, size=(N, N))
    h=0.1
    
    # 1. Ising Interaction Term (J0 * sum J_ij * sigma_z(i) * sigma_z(j))
    for i in range(N):
        for j in range(i + 1, N):
            # Term for sigma_z(i) * sigma_z(j)
            sigma_z_i = get_n_qubit_op(sigma_z, i, N)
            sigma_z_j = get_n_qubit_op(sigma_z, j, N)
             
            H += J_mat[i, j] * sigma_z_i * sigma_z_j
            # H += sigma_z_i * sigma_z_j
            
    # 2. Transverse Field Term (sum h_x(i) * sigma_x(i))
    for i in range(N):
        sigma_x_i = get_n_qubit_op(sigma_x, i, N)
        H += h * sigma_x_i
    return H

H = build_hamiltonian(N)
# --- 4. Reservoir Dynamics Step ---

def qrc_step(rho_k, H, s_k, delta_t):
    """
    Performs one step of QRC: 
    1. Unitary Evolution
    2. Input Injection (CPTP Map)
    """
    
    # --- 1. Unitary Evolution ---
    # Solve the Schrodinger/Von Neumann equation for one time step
    # H is time-independent for this step, so we use matrix exponentiation (equivalent to mesolve without collapse operators)
    U_delta_t = (-1j * H * delta_t / hbar).expm()
    rho_t_plus_delta = U_delta_t * rho_k * U_delta_t.dag()
    
    # --- 2. Input Injection (CPTP Map) ---
    
    # a. Input Encoding: s_k -> |psi_s_k>
    # We choose a simple encoding: s_k maps to an angle theta_k = pi * s_k
    theta_k = np.pi * s_k
    
    # |psi_s_k> = cos(theta_k/2)|0> + sin(theta_k/2)|1>
    # We use QuTiP's basis vector: |0> is basis(2, 0), |1> is basis(2, 1)
    psi_s_k = np.cos(theta_k/2) * basis(2, 0) + np.sin(theta_k/2) * basis(2, 1)
    rho_input = psi_s_k * psi_s_k.dag() # Input density matrix (2x2)
    
    # b. Partial Trace (Remove Qubit 1)
    # The input qubit (index 0) is separated from the reservoir (indices 1 to N-1)
    # The first qubit (index 0) is the one receiving the input.
    qubit_index_to_trace = 0 
    
    # rho_prime = Tr_1(rho_t_plus_delta)
    # In QuTiP, partial_trace traces out the specified indices.
    # The reservoir part (rho_prime) consists of qubits 1 to N-1
    indices_to_keep = list(range(1, N))
    rho_prime = rho_t_plus_delta.ptrace(indices_to_keep) 
    
    # c. Tensor Product (New state is |psi_s_k> <psi_s_k| \otimes rho')
    rho_next = tensor(rho_input, rho_prime)
    
    return rho_next


def obs(input_sequence):
    initial_state = tensor([basis(2, 0)] * N)
    rho_initial = initial_state * initial_state.dag()
    
    
    
    # # Input sequence (a simple sequence of two steps)
    # input_sequence = [0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    
    observables = np.zeros((N,len(input_sequence)))
    
    rho_current = rho_initial
    # print("\n--- Simulation Trace (2 Steps) ---")
    
    for k, s_k in enumerate(input_sequence):
        # print(f"Step {k+1}: Injecting input s_k = {s_k}")
        
        for j in range(N):
            rho_next = qrc_step(rho_current, H, s_k, delta_t)
            
            # Readout for QRC (part of the dynamics, often used to check state)
            # Readout for first qubit: <sigma_z(1)>
            
            readout_op = get_n_qubit_op(sigma_z, j, N)
            readout_value = expect(rho_next,readout_op)
            rho_current = rho_next
            observables[j,k] = readout_value
            
            
    observables = np.reshape(observables,N*len(input_sequence))   
    # observables = np.reshape(observables,(N,50))
    return observables




