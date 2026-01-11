import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time


# ============================================================
# Day 9 SETTINGS (simple knobs)
# ============================================================
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Training steps (keep these modest for a Day-9 session)
PRETRAIN_STEPS = 300          # your warmup
POST_STEPS     = 500          # training after adaptive refresh
PRINT_EVERY    = 100

# Adaptive sampling knobs
N_CAND     = 8000
K_REPLACE  = 800
K_LOCAL    = 200
CAND_BATCH = 512
SIGMA      = 0.03

# Mixture knobs (Day 9 main feature)
ALPHA     = 0.7      # 70% exploit (bad points) + 30% explore (random)
POWER     = 1.5      # >1 means stronger bias toward worst points
POOL_MULT = 10       # exploit pool size = POOL_MULT * K

# Stability knobs
LR        = 5e-3
CLIP_NORM = 5.0

# Loss balancing (Day 9 feature)
USE_EMA_WEIGHTS = True  # set False to go back to your fixed lambdas
EMA_BETA = 0.99

print("TensorFlow:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices("GPU")) > 0)

# ============================================================
# --- PDE setup (your exact PDE)
# True solution: u(x,y)=sin(pi x) sin(pi y)
# PDE: ∇²u + 2π² u = 0
# scaled residual: (∇²u/(2π²)) + u = 0
# ============================================================
pi = tf.constant(np.pi, tf.float32)
c  = tf.constant(2.0*(np.pi**2), tf.float32)  # 2*pi^2

def u_true_xy(X):  # X: (N,2)
    x = X[:,0:1]
    y = X[:,1:2]
    return tf.sin(pi*x) * tf.sin(pi*y)

# ============================================================
# --- Data points (supervised)
# ============================================================
N_u = 400
X_u = tf.random.uniform((N_u, 2), -1.0, 1.0, dtype=tf.float32)
u_u = u_true_xy(X_u)

# ============================================================
# --- Boundary points
# NOTE: your code makes 4*N_b boundary points (x=±1 and y=±1)
# ============================================================
N_b = 250
t = tf.random.uniform((N_b, 1), -1.0, 1.0, dtype=tf.float32)
X_b = tf.concat([
    tf.concat([ tf.ones_like(t),  t], axis=1),   # x= 1
    tf.concat([-tf.ones_like(t),  t], axis=1),   # x=-1
    tf.concat([ t,  tf.ones_like(t)], axis=1),   # y= 1
    tf.concat([ t, -tf.ones_like(t)], axis=1),   # y=-1
], axis=0)
u_b = u_true_xy(X_b)

# ============================================================
# --- Physics pool (collocation)
# ============================================================
N_f = 2500
X_f = tf.random.uniform((N_f, 2), -1.0, 1.0, dtype=tf.float32)

print("X_u:", X_u.shape, "u_u:", u_u.shape)
print("X_b:", X_b.shape, "u_b:", u_b.shape)
print("X_f:", X_f.shape)

# ============================================================
# --- Model (your exact model)
# ============================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(1)
])

# ============================================================
# Block 0 helper: MSE + batch sampler (your style)
# ============================================================
def mse(a, b):
    return tf.reduce_mean(tf.square(a - b))

def sample_batch(X, y=None, batch=128):
    idx = tf.random.uniform((batch,), 0, tf.shape(X)[0], dtype=tf.int32)
    if y is None:
        return tf.gather(X, idx)
    return tf.gather(X, idx), tf.gather(y, idx)

# ============================================================
# Day 9 Upgrade 1: FAST u_and_lap (no retracing)
# BUT still callable as u_and_lap(model, X) (same signature)
# ============================================================

def _make_u_and_lap_compiled(model):
    # This compiled function accepts X with shape [None, 2] (any batch size, 2 columns)
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)],
        reduce_retracing=True
    )
    def u_and_lap_X(X):
        X = tf.cast(X, tf.float32)

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(X)
            with tf.GradientTape() as tape1:
                tape1.watch(X)
                u = model(X)  # (N,1)

            du_dX = tape1.gradient(u, X)  # (N,2)
            ux = du_dX[:, 0:1]            # (N,1)
            uy = du_dX[:, 1:2]            # (N,1)

        # second derivatives one by one (less peak memory)
        g = tape2.gradient(ux, X)        # (N,2)
        uxx = g[:, 0:1]
        del g

        g = tape2.gradient(uy, X)        # (N,2)
        uyy = g[:, 1:2]
        del g

        del tape2

        lap = uxx + uyy
        return u, lap

    return u_and_lap_X

_u_and_lap_X = _make_u_and_lap_compiled(model)

# IMPORTANT: keep your original signature: u_and_lap(model, X)
def u_and_lap(model_unused, X):
    # model_unused is ignored; model is already captured in _u_and_lap_X
    return _u_and_lap_X(X)

# ============================================================
# Tail stats + fixed grid eval (Day 9 instrumentation)
# ============================================================
def residual_abs(X):
    u, lap = u_and_lap(model, X)
    res = (lap / c) + u
    return tf.abs(res)[:, 0]  # (N,)

def tail_stats(r_abs, name="|res|"):
    r = tf.reshape(r_abs, [-1]).numpy()
    r = np.abs(r)
    p = np.percentile(r, [50, 75, 90, 95, 99, 99.5, 99.9])
    top10 = np.sort(r)[-10:][::-1]
    print(f"{name}: median={p[0]:.4g}, p90={p[2]:.4g}, p99={p[4]:.4g}, p99.9={p[-1]:.4g}, max={r.max():.4g}")
    print("top10:", top10)

def eval_fixed_grid(n=121):
    xs = np.linspace(-1.0, 1.0, n).astype(np.float32)
    Xg = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    Xg = tf.constant(Xg, tf.float32)

    r_abs = residual_abs(Xg)
    maxv = float(tf.reduce_max(r_abs))
    rmsv = float(tf.sqrt(tf.reduce_mean(tf.square(r_abs))))

    # also compute u error vs true (nice sanity check)
    u_pred = model(Xg)
    u_true = u_true_xy(Xg)
    l2_u = float(tf.sqrt(tf.reduce_mean(tf.square(u_pred - u_true))))

    return maxv, rmsv, l2_u, r_abs

# ============================================================
# Day 9 Upgrade 2: Mixture Adaptive Sampling (batched scoring)
# Keeps the same "style" as your v3.
# ============================================================
def adaptive_update_Xf_batched_v4_mixture(
    model, X_f, u_and_lap, c,
    N_cand=N_CAND, K=K_REPLACE, k_local=K_LOCAL,
    sigma=SIGMA, cand_batch=CAND_BATCH,
    alpha=ALPHA, power=POWER, pool_mult=POOL_MULT,
    eps=1e-8
):
    # 1) candidate pool
    X_cand = tf.random.uniform((N_cand, 2), -1.0, 1.0, dtype=tf.float32)

    vals_list = []
    idx_list  = []

    # 2) score candidates in batches
    for start in range(0, N_cand, cand_batch):
        end = min(N_cand, start + cand_batch)
        X_batch = X_cand[start:end]

        u_c, lap_c = u_and_lap(model, X_batch)
        score = tf.abs((lap_c / c) + u_c)[:, 0]  # (batch,)

        kb = min(k_local, end - start)
        vals_b, idx_b = tf.math.top_k(score, k=kb)
        idx_b = idx_b + start

        vals_list.append(vals_b)
        idx_list.append(idx_b)

    all_vals = tf.concat(vals_list, axis=0)
    all_idx  = tf.concat(idx_list,  axis=0)

    total_kept = int(all_vals.shape[0])
    if total_kept == 0:
        return X_f, 0.0, 0.0, 0, 0

    # reporting stats like v3: top-K among merged local tops
    K_eff = min(K, total_kept)
    best_vals, _ = tf.math.top_k(all_vals, k=K_eff)
    max_top  = float(best_vals[0])
    mean_top = float(tf.reduce_mean(best_vals))

    # 3) build exploit pool
    M_eff = min(pool_mult * K_eff, total_kept)
    pool_vals, pool_pick = tf.math.top_k(all_vals, k=M_eff)
    pool_idx = tf.gather(all_idx, pool_pick)

    # 4) exploit/explore counts
    K_exploit = int(round(alpha * K_eff))
    K_exploit = max(0, min(K_exploit, M_eff))
    K_explore = K_eff - K_exploit

    # 5A) exploit: soft-top without replacement (Gumbel-top-k)
    if K_exploit > 0:
        logits = power * tf.math.log(pool_vals + eps)
        u = tf.random.uniform(tf.shape(logits), minval=1e-6, maxval=1.0-1e-6, dtype=logits.dtype)
        g = -tf.math.log(-tf.math.log(u))
        noisy = logits + g
        _, exploit_pick = tf.math.top_k(noisy, k=K_exploit)
        exploit_idx = tf.gather(pool_idx, exploit_pick)
    else:
        exploit_idx = tf.constant([], dtype=tf.int32)

    # 5B) explore: uniform random from full candidate pool
    if K_explore > 0:
        explore_idx = tf.random.shuffle(tf.range(N_cand))[:K_explore]
    else:
        explore_idx = tf.constant([], dtype=tf.int32)

    chosen_idx = tf.concat([exploit_idx, explore_idx], axis=0)

    # 6) gather + jitter + clip
    X_sel = tf.gather(X_cand, chosen_idx)
    X_new = X_sel + sigma * tf.random.normal(tf.shape(X_sel), dtype=tf.float32)
    X_new = tf.clip_by_value(X_new, -1.0, 1.0)

    # 7) replace points in X_f (keep size fixed)
    Nf = tf.shape(X_f)[0]
    keep_n = tf.maximum(Nf - K_eff, 0)
    keep = tf.random.shuffle(tf.range(Nf))[:keep_n]
    X_keep = tf.gather(X_f, keep)
    X_f_new = tf.concat([X_keep, X_new], axis=0)

    return X_f_new, max_top, mean_top, K_exploit, K_explore

# ============================================================
# Day 9 Upgrade 3: EMA loss balancing + gradient clipping
# ============================================================

# Your original fixed weights (used only if USE_EMA_WEIGHTS=False)
lambda_phys = tf.constant(2e-1, tf.float32)
lambda_bc   = tf.constant(1.0, tf.float32)
lambda_data = tf.constant(1.0, tf.float32)

ema = {
    "data": tf.Variable(1.0, trainable=False, dtype=tf.float32),
    "bc":   tf.Variable(1.0, trainable=False, dtype=tf.float32),
    "pde":  tf.Variable(1.0, trainable=False, dtype=tf.float32),
}
beta = tf.constant(EMA_BETA, tf.float32)
eps  = tf.constant(1e-8, tf.float32)

def update_ema(key, val):
    ema[key].assign(beta * ema[key] + (1.0 - beta) * tf.stop_gradient(val))

def get_weights(Ld, Lb, Lp):
    if not USE_EMA_WEIGHTS:
        return lambda_data, lambda_bc, lambda_phys

    update_ema("data", Ld)
    update_ema("bc",   Lb)
    update_ema("pde",  Lp)

    w_d = 1.0 / (ema["data"] + eps)
    w_b = 1.0 / (ema["bc"]   + eps)
    w_p = 1.0 / (ema["pde"]  + eps)
    s = w_d + w_b + w_p
    return w_d/s, w_b/s, w_p/s

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

# Train step (still eager, simple, and always uses latest X_f)
def train_step_once():
    global X_f

    Xu, uu = sample_batch(X_u, u_u, batch=128)
    Xb, ub = sample_batch(X_b, u_b, batch=256)
    Xf_b   = sample_batch(X_f, None, batch=512)

    with tf.GradientTape() as tape:
        Ld = mse(uu, model(Xu))
        Lb = mse(ub, model(Xb))

        u_f, lap_f = u_and_lap(model, Xf_b)
        res = (lap_f / c) + u_f
        Lp = tf.reduce_mean(tf.square(res))

        w_d, w_b, w_p = get_weights(Ld, Lb, Lp)
        L = w_d*Ld + w_b*Lb + w_p*Lp

    grads = tape.gradient(L, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, CLIP_NORM)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return L, Ld, Lp, Lb, w_d, w_b, w_p

# ============================================================
# ===================== DAY 9 RUN ============================
# ============================================================

# --------------------------
# Block 1 (10 min): Baseline stats (before any training)
# --------------------------
print("\n====================")
print("Block 1 (10 min): Baseline stats (before training)")
print("====================")
max0, rms0, l2u0, r0 = eval_fixed_grid(n=121)
print(f"Baseline grid: max|res|={max0:.6f}, rms(|res|)={rms0:.6f}, L2(u)={l2u0:.6f}")
tail_stats(r0, name="Baseline grid |res|")

# --------------------------
# Block 2 (15 min): Warm-up training (your pretrain)
# --------------------------
print("\n====================")
print("Block 2 (15 min): Warm-up training")
print("====================")
t0 = time.time()
for step in range(PRETRAIN_STEPS):
    L, Ld, Lp, Lb, wd, wb, wp = train_step_once()
    if step % PRINT_EVERY == 0:
        print(f"pretrain {step:4d} | total {float(L):.6f} | data {float(Ld):.6f} | phys {float(Lp):.6f} | bc {float(Lb):.6f} | w(d,bc,p)=({float(wd):.3f},{float(wb):.3f},{float(wp):.3f})")
print("Warm-up time (s):", round(time.time() - t0, 2))

print("\nStats after warm-up (before adaptive refresh):")
max_w, rms_w, l2u_w, r_w = eval_fixed_grid(n=121)
print(f"Warm-up grid: max|res|={max_w:.6f}, rms(|res|)={rms_w:.6f}, L2(u)={l2u_w:.6f}")
tail_stats(r_w, name="Warm-up grid |res|")

# --------------------------
# Block 3 (20 min): Adaptive refresh (mixture) + train after refresh
# --------------------------
print("\n====================")
print("Block 3 (20 min): Adaptive refresh (mixture) + training")
print("====================")

print("Before X_f:", X_f.shape)
X_f, max_top, mean_top, k_exp, k_expl = adaptive_update_Xf_batched_v4_mixture(
    model=model, X_f=X_f, u_and_lap=u_and_lap, c=c,
    N_cand=N_CAND, K=K_REPLACE, k_local=K_LOCAL,
    cand_batch=CAND_BATCH, sigma=SIGMA,
    alpha=ALPHA, power=POWER, pool_mult=POOL_MULT
)
print("After  X_f:", X_f.shape)
print(f"Exploit points: {k_exp} | Explore points: {k_expl}")
print(f"Top-K (like your v3): Max top-K |res|={max_top:.6f}, Mean top-K|res|={mean_top:.6f}")

print("\nTraining after refresh...")
t0 = time.time()
for step in range(POST_STEPS):
    L, Ld, Lp, Lb, wd, wb, wp = train_step_once()
    if step % PRINT_EVERY == 0:
        print(f"post {step:4d} | total {float(L):.6f} | data {float(Ld):.6f} | phys {float(Lp):.6f} | bc {float(Lb):.6f} | w(d,bc,p)=({float(wd):.3f},{float(wb):.3f},{float(wp):.3f})")
print("Post-train time (s):", round(time.time() - t0, 2))

# --------------------------
# Block 4 (15 min): Final stats + compare
# --------------------------
print("\n====================")
print("Block 4 (15 min): Final stats + comparison")
print("====================")
max1, rms1, l2u1, r1 = eval_fixed_grid(n=121)
print(f"Final grid: max|res|={max1:.6f}, rms(|res|)={rms1:.6f}, L2(u)={l2u1:.6f}")
tail_stats(r1, name="Final grid |res|")

print("\nImprovement summary (grid):")
print(f"max|res|: {max_w:.6f} -> {max1:.6f}  (delta {max1-max_w:+.6f})")
print(f"rms|res|: {rms_w:.6f} -> {rms1:.6f}  (delta {rms1-rms_w:+.6f})")
print(f"L2(u)  : {l2u_w:.6f} -> {l2u1:.6f}  (delta {l2u1-l2u_w:+.6f})")

# Optional: quick plot of prediction vs true (simple visual check)
# (Keeps it lightweight; you can remove if you don't want plots.)
def plot_pred_vs_true(n=101):
    xs = np.linspace(-1.0, 1.0, n).astype(np.float32)
    Xg = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    Xg = tf.constant(Xg, tf.float32)

    u_p = model(Xg).numpy().reshape(n, n)
    u_t = u_true_xy(Xg).numpy().reshape(n, n)
    err = np.abs(u_p - u_t)

    plt.figure()
    plt.title("u_pred")
    plt.imshow(u_p, extent=[-1,1,-1,1], origin="lower")
    plt.colorbar()

    plt.figure()
    plt.title("u_true")
    plt.imshow(u_t, extent=[-1,1,-1,1], origin="lower")
    plt.colorbar()

    plt.figure()
    plt.title("|u_pred - u_true|")
    plt.imshow(err, extent=[-1,1,-1,1], origin="lower")
    plt.colorbar()
    plt.show()

# Uncomment if you want the plots:
# plot_pred_vs_true(n=101)
