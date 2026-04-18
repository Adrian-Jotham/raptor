# RAPTOR — Foundation Policy for Quadrotor Control

RAPTOR trains a **universal foundation policy** — a single small neural network that can control multiple quadrotor drone types (Crazyflie, X500, MRS, race drones, etc.) without hardware-specific tuning. It runs at 100 Hz directly on embedded flight controllers (PX4, Betaflight, Crazyflie, M5StampFly).

The approach: train 1000 independent expert policies via Reinforcement Learning on diverse simulated dynamics, then distill them all into one compact recurrent student policy via behavioral cloning.

---

## Table of Contents

- [How It Works (High-Level)](#how-it-works-high-level)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
  - [Phase 1 — Pre-Training (SAC)](#phase-1--pre-training-sac)
  - [Phase 2 — Post-Training (Behavioral Cloning + DAgger)](#phase-2--post-training-behavioral-cloning--dagger)
- [Neural Network Architectures](#neural-network-architectures)
- [Observation Space](#observation-space)
- [Tunable Parameters](#tunable-parameters)
  - [Phase 1 — SAC / Network](#phase-1--sac--network)
  - [Phase 1 — Domain Randomization](#phase-1--domain-randomization)
  - [Phase 1 — Reward Function](#phase-1--reward-function)
  - [Phase 1 — Options / Feature Flags](#phase-1--options--feature-flags)
  - [Phase 2 — Student / Training](#phase-2--student--training)
  - [Build Options](#build-options)
- [Running the Pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Supported Hardware Platforms](#supported-hardware-platforms)

---

## How It Works (High-Level)

```
Diverse simulated drones (1000 configs)
        │
        ▼
┌─────────────────────┐
│  Phase 1: SAC RL    │  ← 1000 independent expert ("teacher") policies
│  (1M steps each)    │    each trained on a specific drone dynamics
└─────────────────────┘
        │ top checkpoints
        ▼
┌─────────────────────┐
│  Phase 2: BC +      │  ← single small GRU student learns to imitate all
│  DAgger distill.    │    teachers, generalizing across all dynamics
└─────────────────────┘
        │
        ▼
  ~2000-parameter GRU policy
  deployable on real hardware
```

The key insight is **extreme domain randomization**: drone mass ranges from 20 g to 5 kg (250×), thrust-to-weight ratio from 1.5× to 5×. This forces the policy to learn the fundamental physics of quadrotor control rather than any hardware-specific behavior.

---

## Project Structure

```
raptor/
├── rl-tools/                          # Core framework (git submodule)
│   ├── CMakeLists.txt                 # Top-level build config
│   ├── src/foundation_policy/
│   │   ├── pre_training/
│   │   │   ├── main.cpp               # SAC training orchestration
│   │   │   ├── config.h               # SAC hyperparameters & network arch
│   │   │   ├── environment.h          # Multirotor simulator spec
│   │   │   ├── options.h              # Feature flags
│   │   │   └── sample_dynamics_parameters.cpp  # Generate 1000 drone configs
│   │   ├── post_training/
│   │   │   ├── main.cpp               # Behavioral cloning + DAgger loop
│   │   │   ├── config.h               # Student network & training params
│   │   │   ├── environment.h          # Post-training env (no noise)
│   │   │   └── helper.h               # Trajectory sampling utilities
│   │   ├── dynamics_parameters/       # 1000 pre-generated JSON drone configs
│   │   ├── registry/                  # Hardware platform definitions
│   │   │                              #   crazyflie, x500, mrs, fs, race, flightmare
│   │   └── CMakeLists.txt
│   ├── include/                       # Header-only RLtools library
│   └── external/                      # Dependencies (highfive, json, tensorboard)
└── data/                              # Pre-trained checkpoint tarballs
```

---

## Training Pipeline

### Phase 1 — Pre-Training (SAC)

**Goal:** Train 1000 independent expert "teacher" policies, one per simulated drone configuration.

**Algorithm:** Soft Actor-Critic (SAC) — off-policy RL with entropy regularization.

**Steps:**

1. **Sample dynamics** — generate 1000 random drone configuration JSON files:
   ```bash
   ./build/src/foundation_policy/foundation_policy_pre_training_sample_dynamics_parameters
   ```
   Each JSON specifies mass, inertia, rotor positions, motor time constants, thrust curves, etc.

2. **Train teachers** — run SAC on each config independently (parallelizable):
   ```bash
   seq 0 999 | xargs -P <num_cores> -I {} \
     ./build/src/foundation_policy/foundation_policy_pre_training \
     ./src/foundation_policy/dynamics_parameters/{}.json
   ```
   Each run: 1M environment steps, 100 Hz simulation, 5-second episodes.

3. **Rank checkpoints** — extract and sort teachers by evaluation return:
   ```bash
   ./extract_checkpoints.sh   # produces checkpoints_{timestamp}.txt
   ```

**What happens during SAC training per seed:**

```
Reset episode → sample target position
  ↓
Observe 26D state (pos, orientation, velocity, prev action, rotor speeds)
  ↓
Actor forward pass → 4D normalized motor command [-1, 1]
  ↓
Multirotor physics step (motor lag + rigid body dynamics)
  ↓
Compute reward (position error + action smoothness)
  ↓
Store in replay buffer
  ↓
Every 2 steps: update critics (Q-function) via TD target
Every 2 steps: update actor to maximize Q + entropy
Every step:   update entropy coefficient α
  ↓
Repeat for 1M steps, checkpoint every 100k steps
```

---

### Phase 2 — Post-Training (Behavioral Cloning + DAgger)

**Goal:** Distill all 1000 teacher policies into a single compact GRU student.

**Algorithm:** Behavioral Cloning (first 10 epochs) → DAgger (remaining 990 epochs).

**Steps:**

1. **Load teachers** — read top 1000 Phase 1 checkpoints listed in `checkpoints_{timestamp}.txt`
2. **Initialize student** — small GRU network (~2000 parameters)
3. **Run training loop** (1000 epochs total):

```
Epochs 0–9 (Behavioral Cloning):
  For each teacher:
    Sample 10 trajectories using teacher policy
    Record (observation, teacher_action) pairs
  Train student: minimize MSE(student_action, teacher_action)

Epochs 10–999 (DAgger):
  Evaluate student on all 1000 dynamics configs
  Select top-performing teachers for this epoch
  For each active teacher:
    Roll out student trajectory
    Query teacher for correct action at each student state
    Add (state, teacher_action) to dataset
  Train student: minimize MSE on augmented dataset
```

DAgger corrects the **distributional shift** problem: after pure behavioral cloning, the student visits different states than the teacher did during data collection. DAgger re-queries teachers on states the student actually encounters.

---

## Neural Network Architectures

### Phase 1 — Teacher Actor (SAC)

```
Input: 26D observation
  → Dense(26 → 64) + ReLU
  → Dense(64 → 64) + ReLU
  → Dense(64 → 8)          [outputs mean + log_std for 4 motors]
  → Tanh squash → 4D action ∈ [-1, 1]

Critic (×2 for stability):
Input: 26D obs + 4D action = 30D
  → Dense(30 → 256) + ReLU
  → Dense(256 → 256) + ReLU
  → Dense(256 → 1)         [scalar Q-value]
```

### Phase 2 — Student Policy (GRU)

```
Input: 22D observation (no trajectory/rotor-speed info)
  → Dense(22 → 16) + ReLU
  → GRU(16 → 16)           [stateful recurrence across 500-step episodes]
  → Dense(16 → 4)           [linear output]
  → Tanh at inference → 4D action ∈ [-1, 1]

Total parameters: ~2000
```

The student uses a **stateful GRU**: it implicitly identifies the drone's dynamics from the trajectory history rather than needing explicit dynamics parameters as input. The GRU hidden state is reset at the start of each episode.

---

## Observation Space

### Phase 1 Teacher (26D)

| Component | Dim | Description |
|-----------|-----|-------------|
| Position | 3 | Absolute XYZ position (m) |
| Orientation | 9 | Flattened rotation matrix |
| Linear velocity | 3 | Body-frame velocity relative to target (m/s) |
| Angular velocity | 3 | Body-frame angular rates (rad/s) |
| Previous action | 4 | Last motor command [-1, 1] |
| Rotor speeds | 4 | Current RPM (normalized) |

### Phase 2 Student (22D)

Same as above minus rotor speeds (4D), minus trajectory info. The student receives no explicit hardware parameters — it must infer them from the GRU hidden state.

---

## Tunable Parameters

### Phase 1 — SAC / Network

File: [rl-tools/src/foundation_policy/pre_training/config.h](raptor/rl-tools/src/foundation_policy/pre_training/config.h)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ACTOR_HIDDEN_DIM` | 64 | Teacher actor hidden layer width |
| `ACTOR_NUM_LAYERS` | 3 | Teacher actor depth |
| `CRITIC_HIDDEN_DIM` | 256 | Critic hidden layer width |
| `CRITIC_NUM_LAYERS` | 3 | Critic depth |
| `STEP_LIMIT` | 1,000,000 | Total training steps per seed |
| `EPISODE_STEP_LIMIT` | 500 | Max steps per episode (5 sec @ 100 Hz) |
| `REPLAY_BUFFER_CAP` | 1,000,000 | Replay buffer size |
| `ACTOR_BATCH_SIZE` | 128 | Actor update batch size |
| `CRITIC_BATCH_SIZE` | 128 | Critic update batch size |
| `ACTOR_TRAINING_INTERVAL` | 2 | Train actor every N environment steps |
| `CRITIC_TRAINING_INTERVAL` | 1 | Train critic every N environment steps |
| `N_WARMUP_STEPS_CRITIC` | 10,000 | Steps before critic training starts |
| `N_WARMUP_STEPS_ACTOR` | 10,000 | Steps before actor training starts |
| `GAMMA` | 0.99 | Discount factor |
| `TARGET_ENTROPY` | -2.0 | SAC entropy target (≈ -action_dim/2) |
| `ACTOR_LR` | 3e-4 | Actor Adam learning rate |
| `CRITIC_LR` | 3e-4 | Critic Adam learning rate |
| `ALPHA_LR` | 1e-4 | Entropy coefficient learning rate |
| `NUM_CHECKPOINTS` | 10 | Number of checkpoints saved per run |
| `NUM_EVALUATIONS` | 100 | Evaluation frequency (every STEP_LIMIT/100 steps) |
| `NUM_EVALUATION_EPISODES` | 100 | Episodes per evaluation |

---

### Phase 1 — Domain Randomization

File: [rl-tools/src/foundation_policy/pre_training/sample_dynamics_parameters.cpp](raptor/rl-tools/src/foundation_policy/pre_training/sample_dynamics_parameters.cpp)

These are the bounds for the randomly sampled drone configurations. Wider ranges = more robust policy.

| Parameter | Min | Max | Notes |
|-----------|-----|-----|-------|
| `mass` | 0.02 kg | 5.0 kg | 250× range |
| `thrust_to_weight` | 1.5 | 5.0 | Hover margin |
| `torque_to_inertia` | 40 rad/s² | 1200 rad/s² | Rotational agility |
| `rotor_time_constant_rising` | 0.03 s | 0.10 s | Motor spin-up speed |
| `rotor_time_constant_falling` | 0.03 s | 0.30 s | Motor spin-down speed |
| `rotor_torque_constant` | 0.005 | 0.05 | Drag-to-thrust ratio |
| `mass_size_deviation` | — | 10% | Inertia asymmetry |
| `disturbance_force_max` | — | 0.3 N | Random wind gust |
| `orientation_offset_angle_max` | — | 0.0 rad | IMU misalignment (off) |

---

### Phase 1 — Reward Function

File: [rl-tools/src/foundation_policy/pre_training/sample_dynamics_parameters.cpp](raptor/rl-tools/src/foundation_policy/pre_training/sample_dynamics_parameters.cpp) → `overwrite()` function

| Parameter | Default | Description |
|-----------|---------|-------------|
| `constant` | 1.5 | Per-step survival reward |
| `termination_penalty` | -100.0 | Penalty for crashing / leaving bounds |
| `position` | 1.0 | Weight for position tracking error |
| `orientation` | 0.1 | Weight for orientation error |
| `linear_velocity` | 0.0 | Velocity penalty (disabled) |
| `angular_velocity` | 0.0 | Angular rate penalty (disabled) |
| `action` | 0.0 | Action magnitude penalty (disabled) |
| `d_action` | 1.0 | Action smoothness penalty (jerk) |
| `position_error_integral` | 0.0 | Integral position error (disabled) |
| `position_clip` | 0.0 | Clip position error above this value (0 = no clip) |

Increasing `orientation` encourages more level flight; increasing `d_action` forces smoother motor commands. The `termination_penalty` strongly discourages crashes.

---

### Phase 1 — Options / Feature Flags

File: [rl-tools/src/foundation_policy/pre_training/options.h](raptor/rl-tools/src/foundation_policy/pre_training/options.h)

| Option | Default | Description |
|--------|---------|-------------|
| `SEQUENTIAL_MODEL` | false | Use MLP (false) or RNN (true) for teacher |
| `MOTOR_DELAY` | true | Enable first-order motor lag dynamics |
| `RANDOMIZE_MOTOR_MAPPING` | false | Randomly swap rotor assignments |
| `RANDOMIZE_THRUST_CURVES` | false | Add noise to motor thrust curves |
| `OBSERVE_THRASH_MARKOV` | false | Include randomization params in observation |
| `SAMPLE_INITIAL_PARAMETERS` | false | Resample dynamics each episode (vs. each run) |

---

### Phase 2 — Student / Training

File: [rl-tools/src/foundation_policy/post_training/config.h](raptor/rl-tools/src/foundation_policy/post_training/config.h)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HIDDEN_DIM` | 16 | GRU hidden size (and dense layer width) |
| `N_EPOCH` | 1000 | Total training epochs |
| `EPOCH_TEACHER_FORCING` | 10 | Epochs of pure behavioral cloning before DAgger |
| `BATCH_SIZE` | 64 | Training batch size |
| `SEQUENCE_LENGTH` | 500 | Episode length fed to GRU during training |
| `NUM_EPISODES` | 10 | Trajectories sampled per teacher per epoch |
| `NUM_EPISODES_EVAL` | 100 | Episodes for student evaluation |
| `NUM_TEACHERS` | 1000 | Total teacher checkpoints loaded |
| `NUM_ACTIVE_TEACHERS` | 1000 | Teachers used per DAgger epoch |
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `TEACHER_DETERMINISTIC` | true | Teachers use mean action (no sampling) |
| `ON_POLICY` | true | Re-sample trajectories each epoch |
| `SHUFFLE` | true | Shuffle episodes within each epoch |
| `SOLVED_RETURN` | 300.0 | Minimum return to include a teacher checkpoint |
| `STEADY_STATE_POSITION_CORRECTION` | true | Subtract mean position offset from student obs |
| `STEADY_STATE_OFFSET_ESTIMATION_START` | 250 | Step at which to begin offset estimation (2.5 s) |

The most impactful student parameters to tune:
- **`HIDDEN_DIM`** — larger = more capacity but slower inference on embedded hardware
- **`EPOCH_TEACHER_FORCING`** — more BC epochs before DAgger can improve stability
- **`N_EPOCH`** — training converges around 300–500 epochs typically
- **`LEARNING_RATE`** — reduce if training is unstable

---

### Build Options

Passed to CMake at configure time:

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Use `Release` for training speed |
| `RL_TOOLS_BACKEND_ENABLE_MKL` | OFF | Enable Intel MKL for fast matrix ops (recommended) |
| `RL_TOOLS_ENABLE_HDF5` | OFF | Required for checkpoint save/load |
| `RL_TOOLS_ENABLE_JSON` | OFF | Required for dynamics parameter loading |
| `RL_TOOLS_ENABLE_TENSORBOARD` | OFF | Enable TensorBoard logging |
| `RL_TOOLS_ENABLE_TARGETS` | OFF | Enable foundation_policy build targets |
| `RL_TOOLS_EXPERIMENTAL` | OFF | Required for foundation_policy targets |

---

## Running the Pipeline

### 1. Build

```bash
cd raptor/rl-tools
mkdir build && cd build

MKL_ROOT=/opt/intel/oneapi/mkl/latest cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DRL_TOOLS_BACKEND_ENABLE_MKL=ON \
  -DRL_TOOLS_ENABLE_TARGETS=ON \
  -DRL_TOOLS_EXPERIMENTAL=ON \
  -DRL_TOOLS_ENABLE_HDF5=ON \
  -DRL_TOOLS_ENABLE_JSON=ON \
  -DRL_TOOLS_ENABLE_TENSORBOARD=ON

cmake --build . \
  --target foundation_policy_pre_training_sample_dynamics_parameters \
  --target foundation_policy_pre_training \
  --target foundation_policy_post_training \
  -j$(nproc)
```

### 2. Phase 1 — Generate dynamics configs

```bash
./build/src/foundation_policy/foundation_policy_pre_training_sample_dynamics_parameters
```

Produces `./src/foundation_policy/dynamics_parameters/{0..999}.json`.

### 3. Phase 1 — Train teachers (parallelizable)

```bash
# Run all 1000 in parallel (adjust -P to your core count)
seq 0 999 | xargs -P 32 -I {} \
  ./build/src/foundation_policy/foundation_policy_pre_training \
  ./src/foundation_policy/dynamics_parameters/{}.json
```

Each run takes ~30–60 minutes depending on hardware. Output goes to `./1k-experiments/`.

### 4. Phase 1 — Extract best checkpoints

```bash
./extract_checkpoints.sh
# produces checkpoints_<timestamp>.txt
```

### 5. Phase 2 — Train student

Edit [post_training/main.cpp](raptor/rl-tools/src/foundation_policy/post_training/main.cpp) to set `checkpoint_path.experiment` to your Phase 1 timestamp, then:

```bash
./build/src/foundation_policy/foundation_policy_post_training
```

Monitor with TensorBoard:
```bash
tensorboard --logdir ./logs
```

---

## Outputs

| Path | Contents |
|------|----------|
| `1k-experiments/foundation-policy-pre-training_<ts>/dynamics-id=N/seed=0/step=M/actor` | Phase 1 teacher checkpoints (HDF5) |
| `1k-experiments/.../return.json` | Evaluation returns for each checkpoint |
| `checkpoints_<ts>.txt` | Ranked list of best Phase 1 checkpoints |
| `logs/<ts>/checkpoints/<epoch>/actor` | Phase 2 student checkpoints (HDF5) |
| `logs/<ts>/actor` | Best Phase 2 student checkpoint |
| `logs/<ts>/events.out.tfevents.*` | TensorBoard training metrics |

TensorBoard metrics logged during Phase 2:
- `loss` — MSE imitation loss per batch
- `evaluation/return/{mean,std}` — student return across all dynamics
- `evaluation/episode_length/{mean,std}` — episode survival time
- `evaluation/share_terminated` — crash rate
- `crazyflie/{return,episode_length,share_terminated}` — Crazyflie-specific eval

---

## Supported Hardware Platforms

Defined in [rl-tools/src/foundation_policy/registry/](raptor/rl-tools/src/foundation_policy/registry/):

| Platform | Description |
|----------|-------------|
| `crazyflie` | Bitcraze Crazyflie 2.1 (30.6 g, 28 mm arm) |
| `x500` | Holybro X500 mid-size quad |
| `mrs` | MRS group platform |
| `fs` | FS platform |
| `race` | Racing drone configuration |
| `flightmare` | Flightmare simulator platform |

The default training base model is `crazyflie`. Phase 2 evaluation uses the Crazyflie dynamics as the canonical benchmark.
