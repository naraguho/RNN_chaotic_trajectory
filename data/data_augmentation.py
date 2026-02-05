import numpy as np
import os

# ==========================================
# Parameters
# ==========================================
base_dir = ""
T_window = 401
save_name_prefix = f"T{T_window}"

# ==========================================
# Loop over all seeds
# ==========================================
for seed in range(1,100):
    seed_dir = os.path.join(base_dir, f"seed_{seed}")
    op_file_Q = os.path.join(seed_dir, "order_parameter_Q.txt")
    op_file_n = os.path.join(seed_dir, "order_parameter_n.txt")

    # Load order parameter data: columns = [time, δQ, δρ]
    data_Q = np.loadtxt(op_file_Q)
    data_n = np.loadtxt(op_file_n)
    
    # trim initial part since it is transient regime before chaotic trajectory
    order_parameter_Q = data_Q[100:] 
    order_parameter_n = data_n[100:]
    # δQ = data[:, 1]
    # δρ = data[:, 2]

    num_steps = len(order_parameter_Q)
    num_samples = num_steps - T_window

    # Prepare directories
    save_dir = os.path.join(seed_dir, "prepared")
    os.makedirs(save_dir, exist_ok=True)

    # ======================================
    # Construct X (input windows), Y (target), T (starting time)
    # ======================================
    X_Q = np.zeros((num_samples, T_window), dtype=np.float64)
    Y_Q = np.zeros(num_samples, dtype=np.float64)
    X_n = np.zeros((num_samples, T_window), dtype=np.float64)
    Y_n = np.zeros(num_samples, dtype=np.float64)
    #T = np.zeros(num_samples, dtype=np.float64)

    for t in range(num_samples):
        X_Q[t, :] = order_parameter_Q[t:t + T_window]
        Y_Q[t] = order_parameter_Q[t + T_window]
        X_n[t, :] = order_parameter_n[t:t + T_window]
        Y_n[t] = order_parameter_n[t + T_window]
    #    T[t] = starting_time[t]

    # Save results
    np.save(os.path.join(save_dir, f"X_order_T{T_window}_Q.npy"), X_Q)
    np.save(os.path.join(save_dir, f"Y_order_T{T_window}_Q.npy"), Y_Q)
    np.savetxt(os.path.join(save_dir, f"X_order_T{T_window}_Q.txt"), X_Q.reshape(num_samples, -1))
    np.savetxt(os.path.join(save_dir, f"Y_order_T{T_window}_Q.txt"), Y_Q)
    
    np.save(os.path.join(save_dir, f"X_order_T{T_window}_n.npy"), X_n)
    np.save(os.path.join(save_dir, f"Y_order_T{T_window}_n.npy"), Y_n)
    np.savetxt(os.path.join(save_dir, f"X_order_T{T_window}_n.txt"), X_n.reshape(num_samples, -1))
    np.savetxt(os.path.join(save_dir, f"Y_order_T{T_window}_n.txt"), Y_n)


    print(f"✅ Seed {seed}: saved {X_Q.shape}, {Y_Q.shape} → {save_dir}")
