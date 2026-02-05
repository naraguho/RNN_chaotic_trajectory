import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import gc


# To demonstration purpose : focused on readability, not optimization for speed. 

# ======================================
# Reproducibility setup
# ======================================
seed = 42  # 42 is used for the previous work
np.random.seed(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

# ======================================
# 1. Load prepared dataset
# ======================================
data_dir = ""
Tlen = 401
X = np.load(os.path.join(data_dir, f"X_train_T{Tlen}_r0.5.npy")) # (# of data, T length, )
Y = np.load(os.path.join(data_dir, f"Y_train_T{Tlen}_r0.5.npy")) # (# of data, )
X_val = np.load(os.path.join(data_dir, f"X_val_T{Tlen}_r0.5.npy")) # (# of data, T length, )
Y_val = np.load(os.path.join(data_dir, f"Y_val_T{Tlen}_r0.5.npy")) # (# of data, )


# ======================================
# 2. Convert to torch tensors
# ======================================
X = torch.tensor(X, dtype=torch.float32)#[:10000] # (demonstartion) : Slice(10000)
Y = torch.tensor(Y, dtype=torch.float32)#[:10000] # (demonstartion) : Slice(10000)

# validation data will be used in the training to see the machine-generated trajectory
X_val = torch.tensor(X_val, dtype=torch.float32) 
Y_val = torch.tensor(Y_val, dtype=torch.float32)

#Training parameter
epochs = 5001 # total training epoch
batch_size = 128
# Checking
print(f"Loaded dataset | X: {X.shape}, Y: {Y.shape}") #verifying data shape
print(f"Epochs: {epochs}, Batch size: {batch_size}") #verifying total epoch and batch size(for generality)

# ======================================
# 3. DataLoaders (shuffle each epoch)
# ======================================
train_dataset = TensorDataset(X, Y) # load training dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=False) # assign data to efficient built loader

# --- Before training loop ---
val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)

# ======================================
# 4. Model definition
# ======================================

dp = 500 # embedding space dimension
nh = 1 # number of head

class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, time_len, dt):
        super().__init__()
        self.d_model = d_model
        self.time_len = time_len
        self.dt = dt

        div_term = torch.exp(
            torch.arange(0, d_model, dtype=torch.float32)
            * (-torch.log(torch.tensor(1000.0)) / d_model)
        )
        self.register_buffer("div_term", div_term)

        # precompute relative-time positional encoding: [1, time_len, d_model]
        t = torch.arange(time_len, dtype=torch.float32) * dt  # [time_len]
        pe = torch.zeros(time_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(t.unsqueeze(1) * div_term[0::2])
        pe[:, 1::2] = torch.cos(t.unsqueeze(1) * div_term[1::2])
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x):
        """
        x : [batch, time_len, d_model]
        """
        # broadcast pe across batch
        return x + self.pe[:, :x.size(1), :x.size(2)]
        

class Cherngroup_transformer(nn.Module): 
    def __init__(self, Tlen, d_p=dp, hidden_dim=1536):
        super().__init__()
        self.encoder = nn.Linear(1, d_p)
        self.posenc = PositionalEncoding(d_model = d_p, time_len = Tlen, dt=0.05) # dt value is for demonstration
        self.attn1 = nn.MultiheadAttention(d_p, num_heads=nh, bias=False, batch_first=True)
        self.norm1a = nn.LayerNorm(d_p)
        self.ff1 = nn.Sequential(nn.Linear(d_p, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, d_p))
        self.norm1b = nn.LayerNorm(d_p)
        self.attn2 = nn.MultiheadAttention(d_p, num_heads=nh, bias=False, batch_first=True)
        self.norm2a = nn.LayerNorm(d_p)
        self.ff2 = nn.Sequential(nn.Linear(d_p, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, d_p))
        self.norm2b = nn.LayerNorm(d_p)
        self.pwff = nn.Sequential(nn.Linear(d_p, 1), nn.ReLU())
        self.global_ff = nn.Sequential(
            nn.Linear(Tlen, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1408),
            nn.ReLU(),
            nn.Linear(1408, 1)
        )
    def forward(self, X):
        X = X.unsqueeze(-1)               # [batch, Tlen, 1]
        X = self.encoder(X)               # [batch, Tlen, d_p]
        X = self.posenc(X)
        a1, _ = self.attn1(X, X, X)
        X = self.norm1a(X + a1)
        f1 = self.ff1(X)
        X = self.norm1b(X + f1)
        a2, _ = self.attn2(X, X, X)
        X = self.norm2a(X + a2)
        f2 = self.ff2(X)
        X = self.norm2b(X + f2)
        X_pw = self.pwff(X)
        X_flat = X_pw.flatten(start_dim=1)
        return self.global_ff(X_flat)

# ======================================
# 5. Training setup
# ======================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Cherngroup_transformer(Tlen = Tlen).to(device)
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-6   
)
criterion = nn.MSELoss()


# (demonstration) leaerning rate schedule
def get_lr(epoch):
    if epoch <= 300:
        return 5e-6
    elif epoch <= 800:
        return 1e-6
    elif epoch <= 1300:
        return 5e-7
    else:
        return 1e-7

# ======================================
# 5.5. Directories for saving results
# ======================================
save_dir = os.path.join(data_dir, "results_oneop_T{}_r0.5_more".format(Tlen))
os.makedirs(save_dir, exist_ok=True)

loss_log = []  # store average loss per epoch

# ======================================
# 6. Training loop with validation rollout + saving
# ======================================
    
train_losses = []
val_losses = []

for epoch in range(1, epochs + 1):
    model.train() # start training
    total_loss = 0.0
    
    lr = get_lr(epoch)
    for g in optimizer.param_groups:
        g['lr'] = lr
    print(f"Epoch {epoch} | LR = {lr:.2e}")
    
    for i, (Xb, Yb) in enumerate(train_loader):
        Xb, Yb = Xb.to(device), Yb.to(device) # embed to device(supposed to GPU)
        optimizer.zero_grad() # standard

        Y_pred = model(Xb) # take output from machine
        Y_pred = Y_pred.squeeze(-1) # data processing
        loss = criterion(Y_pred, Yb) # compute MSE loss
        loss.backward() #standard
        optimizer.step() #standard
        total_loss += loss.item() # 

        #only if needed : release mini-batch memory
        #del Xb, Tb, Yb, Y_pred, loss  
        #gc.collect()
        #torch.cuda.empty_cache()

    # ===== Train loss logging =====
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch:03d} | Train Loss = {avg_loss:.6e}")

    # ===== Validation every 5 epochs =====
    if epoch % 5 == 0:
        
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for Xv, Yv in val_loader:
                Xv, Yv = Xv.to(device), Yv.to(device) 
                Yv_pred = model(Xv).squeeze(-1)
                loss = criterion(Yv_pred, Yv)
                val_loss_sum += loss.item() * len(Yv)
                n_val += len(Yv)
    
                #del Xv, Tv, Yv, Yv_pred, loss
                #torch.cuda.empty_cache()
    
        val_loss = val_loss_sum / n_val
        val_losses.append(val_loss)
        print(f"💡 Validation @ Epoch {epoch:03d} | Val Loss = {val_loss:.6e}")
    
    
        # ==== Plot first Trajectory ====
        chunk_size = 833 # For each trajectory(1234), 401 is used to initial input, so 833 is remainder to compare and generate trajectory
        Xv, Yv = X_val[:chunk_size].to(device), Y_val[:chunk_size].to(device)
        Yv_pred = model(Xv).squeeze(-1)
        Yv_np = Yv.cpu().numpy()
        Yv_pred_np = Yv_pred.cpu().detach().numpy()
    
        # ==== Autoregressive machine-generated trajectory ====
        chunk_size = chunk_size
        Xv_init = Xv[0].clone()          # initial input window
        preds_machine = []
        Xv_roll = Xv_init.clone()
    
        model.eval()
        with torch.no_grad():
            for step in range(chunk_size):
                y_pred = model(Xv_roll.unsqueeze(0)).squeeze().item()
                preds_machine.append(y_pred)
                # update rolling window
                Xv_roll = torch.cat([Xv_roll[1:], torch.tensor([y_pred], device=device)], dim=0)
    
        preds_machine_np = np.array(preds_machine)
    
        # ======================================
        # Plot all three (true, direct, autoregressive)
        # ======================================
        plt.figure(figsize=(10, 6))
        plt.plot(range(chunk_size), Yv_np, 'o-', label='Ground Truth $Y_{val}$', alpha=0.8)
        plt.plot(range(chunk_size), Yv_pred_np, 's--', label='Direct Model Output', alpha=0.8)
        plt.plot(range(chunk_size), preds_machine_np, '^-', label='Autoregressive Machine Trajectory', alpha=0.8)
        plt.xlabel("remaining time for one trajectory ")
        plt.ylabel("Output Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
    
        fig_path = os.path.join(save_dir, f"val_{epoch:03d}.png")
        plt.savefig(fig_path, dpi=300)
        plt.show()
        plt.close()
        print(f"📊 Saved validation comparison plot → {fig_path}")
    
        # ======================================
        # Save machine trajectory data
        # ======================================
        traj_dir = os.path.join(save_dir, f"machine_rollout_epoch_{epoch:03d}")
        os.makedirs(traj_dir, exist_ok=True)
    
        np.savetxt(os.path.join(traj_dir, "Y_true.txt"), Yv_np, fmt="%.6f")
        np.savetxt(os.path.join(traj_dir, "Y_direct.txt"), Yv_pred_np, fmt="%.6f")
        np.savetxt(os.path.join(traj_dir, "Y_machine_rollout.txt"), preds_machine_np, fmt="%.6f")
    
        print(f"💾 Saved machine trajectory data → {traj_dir}")
    
        # ==== Save loss logs ====
        np.savetxt(os.path.join(save_dir, "train_loss_log.txt"), np.array(train_losses))
        np.savetxt(os.path.join(save_dir, "val_loss_log.txt"), np.array(val_losses))
    # Save machine
    if epoch % 50 == 0:
        model_path = os.path.join(save_dir, f"model_final_T{Tlen}_dp{dp}_nh{nh}_epoch{epoch}_batch{batch_size}.pt")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }, model_path)



    
