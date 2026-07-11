# DOOM-Agent
DOOM Agent based on SIMA 2 and CLIP

## Project

**Long-Horizon Instruction-Following Multimodal Agent** for 3D game environments.

The agent interprets a long-horizon plan by decomposing it into natural-language sub-goals
(e.g., "find the red door", "pick up the key") and executes them sequentially.

**Success metrics:**
1. Generalization to unseen maps
2. Performance — how reliably and efficiently the agent completes each sub-goal

Primary environment: **ViZDoom**.
Secondary environment: **Unity** (via DoomLoader).
Architecture inspired by **SIMA 2** (Google DeepMind).

---

## Repo Structure

```
DOOM-Agent/
├── doom_agent/
│   ├── agent/
│   │   ├── __init__.py
│   │   └── agent.py              # Agent class — coordinate run loop
│   ├── encoder/
│   │   ├── __init__.py
│   │   └── clip_encoder.py       # FROM CV PROJECT — copy-paste reuse (see below)
│   ├── environment/
│   │   ├── __init__.py
│   │   └── vizdoom_env.py        # ViZDoom wrapper + HUD collection
│   ├── memory/
│   │   ├── __init__.py
│   │   └── buffer.py             # Sliding window buffer for timestep vectors
│   ├── model/
│   │   ├── __init__.py
│   │   ├── temporal_transformer.py  # TransformerEncoder with causal masking
│   │   └── policy_head.py           # Maps transformer output to actions
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py          # Scripted play trajectory collection
│   │   └── relabeler.py          # Hindsight instruction relabeling
│   ├── training/
│   │   ├── __init__.py
│   │   ├── bc_trainer.py         # Behavior cloning training loop
│   │   └── ppo_trainer.py        # PPO via Stable Baselines3
│   ├── tools/
│   │   ├── __init__.py
│   │   └── config.py             # CLI arg parsing
│   └── main.py                   # Entry point
├── artifacts/                    # Checkpoints, logs, tensorboard — gitignored
├── scenarios/                    # ViZDoom .cfg and .wad files
├── Makefile
├── reqs.txt                      # pip dependencies — add all new packages here
├── README.md
```

---

## Running

```bash
make setup          # Create venv and install reqs.txt (python3.11)
make agent          # Run default agent  (--agent=a)
make debug_agent    # Run with debug     (--agent=dt)
make train_agent    # Run training       (--agent=t)
make clean          # Remove venv
```

Custom ARGS passthrough:
```bash
make main ARGS="--agent=dt"
```

---

## CLI Flags (`--agent=<letters>`)

| Flag | Effect |
|------|--------|
| `a`  | Default agent run |
| `d`  | Enable debug output |
| `t`  | Enable training mode |

Flags are combinable: `--agent=dt` = debug + training. Parsed in `doom_agent/tools/config.py`.

---

## Architecture

```
Raw Frame (RGB) ──► [Frozen CLIP Image Encoder] ──► frame embedding
Natural-language sub-goal ──► [Frozen CLIP Text Encoder] ──► goal embedding
HUD values (health / ammo / armor) ──► numeric vector (ViZDoom API — no pixel parsing)

Per-timestep vector = concat(frame_emb, goal_emb, hud_vec)
                                  │
                    [Temporal Transformer]
                    - Sliding window over recent frames
                    - Causal masking during training
                                  │
                         [Policy Head]
                    Phase 1: Behavior Cloning on hindsight-relabeled demo trajectories
                    Phase 2: PPO fine-tuning via Stable Baselines3
```

**CLIP boundary rule:** All new code operates on embeddings *produced by* CLIP.
The encoder, tokenizer, and preprocessing are frozen and must not be modified.

**Text embedding role:** Relevance filter that biases attention toward goal-relevant
visual changes per timestep. It does not drive temporal relationship discovery.

---

## Training Pipeline (ViZDoom)

1. **Collect raw demo trajectories** from scripted play.
2. **Hindsight instruction relabeling** converts trajectories into instruction-labeled
   training data — this addresses the data-generation bottleneck.
3. **Behavior cloning** on labeled data provides a warm start for the policy.
4. **PPO fine-tuning** (Stable Baselines3) improves the policy through environment interaction.

**Reward structure:**
- Dense shaping toward the current active sub-goal
- Sparse completion bonus when the sub-goal is achieved

---

## Multimodal Observation Space

**Core inputs (always):**
- RGB frame → CLIP image embedding
- Active natural-language sub-goal → CLIP text embedding
- Parsed HUD values (health, ammo, armor) as numeric vector

**Why numeric HUD instead of pixels:** Lets the agent reason about concepts like
"health is low → seek health pack" or "ammo is low → switch weapons" without
needing to visually decode the HUD.

**CLIP grounding test:** Whether the agent recognizes "red door" as a portable concept
it can locate on any map, rather than a memorized path on a fixed map.

**Additional goals (depending on project progress):**
- Auto-map / mini-map understanding
- Inventory state understanding

---

## Environment Approaches

### ViZDoom (Primary)
- Train and evaluate on fixed maps
- Maze navigation as the core task
- Establish baseline performance metrics here first

### Unity (Secondary)
- Load Doom maps into Unity via **DoomLoader**
- Compare agent performance against the ViZDoom baseline (no retraining initially)
- Generate additional mazes procedurally for generalization testing
- Procedural generation options: traditional approaches or **DoomGAN**
- Depending on progress: apply the same RL training pipeline in Unity

**Generalization evaluation:** Train in ViZDoom → test same agent in Unity (no retraining)
→ test on new procedurally generated Unity maps.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Game environment (primary) | `vizdoom` |
| Game environment (secondary) | Unity + DoomLoader |
| CLIP | `open_clip_torch` |
| RL training | `stable-baselines3` |
| Deep learning | `torch` |
| Python | 3.11 |

Local development uses MPS (Mac M5 Pro).
Cloud training uses CUDA (RunPod / Vast.ai / Lambda Labs).

---

## Key References

Architecture decisions and findings are from these papers:

| Reference | Relevance |
|-----------|-----------|
| [SIMA 2: A Generalist Embodied Agent for Virtual Worlds](https://deepmind.google/research/publications/sima/) | Primary architecture inspiration |
| [Playing Doom with SLAM-Augmented RL](https://ar5iv.labs.arxiv.org/html/1612.00380) | ViZDoom environment reference |
| [DoomLoader](https://github.com/ChrisHoopes/DoomLoader) | Unity Doom map loading |
| [ViZDoom Environments](https://vizdoom.farama.org/environments/default/) | Environment configs and scenario docs |
| [STEVE-1](https://arxiv.org/abs/2306.00937) | Closest architectural relative; cited, not extended |
| [MineDojo / MineCLIP](https://minedojo.org/) | Multimodal RL in open-world games |
| [VPT (Video PreTraining)](https://arxiv.org/abs/2206.11795) | Behavior cloning from video at scale |
| [DoomGAN (Giacomello et al.)](https://arxiv.org/abs/1806.04726) | GAN-based Doom level generation |
| [PCGRL](https://arxiv.org/abs/2001.09212) | Procedural content generation via RL |

---

## Implementation Plan

Phases are ordered by dependency. Complete each before starting the next.
Each phase maps to one or more modules in the repo structure above.

---

### Phase 1 — ViZDoom Environment Wrapper
**Module:** `doom_agent/environment/vizdoom_env.py`

1. Install `vizdoom` and add to `reqs.txt`.
2. Load a default ViZDoom scenario (start with `basic.cfg` from `scenarios/`).
3. Implement a game loop: `reset()`, `step(action)`, `close()`.
4. Collect HUD values each step using the ViZDoom API — no pixel parsing:
   ```python
   game.get_game_variable(GameVariable.HEALTH)
   game.get_game_variable(GameVariable.AMMO2)
   game.get_game_variable(GameVariable.ARMOR)
   ```
5. Return the RGB frame as a numpy array alongside HUD values.
6. Verify the loop runs for a full episode without errors.

**Done when:** You can step through a full episode and print `(frame.shape, health, ammo, armor)` each step.

---

### Phase 2 — CLIP Encoder Integration
**Module:** `doom_agent/encoder/clip_encoder.py`

1. Copy the encoder setup from the CV project.
2. Adapt `preprocess()` to accept a numpy frame array (not a file path).
3. Add `encode_frame(frame_array) → tensor[512]`.
4. Add `encode_subgoal(text: str) → tensor[512]`.
5. Confirm both return the right shapes and that gradients are disabled.

**Done when:** `encode_frame(frame)` and `encode_subgoal("find the red door")` return `[512]` tensors without errors.

---

### Phase 3 — Observation Assembly + Sliding Window Buffer
**Module:** `doom_agent/memory/buffer.py`

1. Normalize HUD values to 0–1 range and package as `tensor[3]`.
2. Concatenate `(frame_emb, goal_emb, hud_vec)` into a `timestep_vec` of shape `[1027]`.
3. Implement a sliding window buffer that holds the last `N` timestep vectors (start with N=8).
4. Buffer should pad with zeros at the start of an episode.
5. Output: `tensor[N, 1027]` — the input to the temporal transformer.

**Done when:** Buffer fills correctly over a full episode and outputs the right shape.

---

### Phase 4 — Temporal Transformer
**Module:** `doom_agent/model/temporal_transformer.py`

1. Implement a `TransformerEncoder` with causal masking.
2. Input: `tensor[N, 1027]` (sliding window from buffer).
3. Apply positional encoding across the window.
4. Apply causal mask so each position can only attend to itself and earlier positions.
5. Output: the final position's hidden state `tensor[hidden_dim]`.

**Done when:** Random input `[8, 1027]` produces output `[hidden_dim]` with no shape errors.

---

### Phase 5 — Policy Head
**Module:** `doom_agent/model/policy_head.py`

1. Define the ViZDoom action space (discrete button combinations).
2. Implement a linear layer mapping `[hidden_dim] → [num_actions]`.
3. Output logits for each action; apply softmax to get a probability distribution.
4. Add `select_action(logits) → action_index`.

**Done when:** Random hidden state input produces a valid action index.

---

### Phase 6 — Demo Trajectory Collection
**Module:** `doom_agent/data/collector.py`

1. Write a scripted policy (rule-based or random) that plays through episodes.
2. Record each step as a tuple: `(frame, subgoal, hud, action_taken)`.
3. Save trajectories to disk (e.g., as lists of dicts in `.pkl` or `.json`).
4. Collect enough episodes to cover the target maze scenario.

**Done when:** Trajectories are saved to disk and can be loaded and inspected.

---

### Phase 7 — Hindsight Instruction Relabeling
**Module:** `doom_agent/data/relabeler.py`

1. Load collected trajectories.
2. For each trajectory, look at what the agent actually reached/achieved.
3. Retroactively assign a natural-language instruction label matching that outcome
   (e.g., agent reached the exit → label that trajectory "reach the exit").
4. Output: relabeled dataset of `(frame, instruction, hud, action)` tuples.

**Done when:** Relabeled dataset has instruction strings attached to each timestep and can be fed into training.

---

### Phase 8 — Behavior Cloning
**Module:** `doom_agent/training/bc_trainer.py`

1. Build a DataLoader over the relabeled trajectory dataset.
2. For each batch: run encoder + buffer + transformer + policy head.
3. Loss: cross-entropy between predicted action logits and recorded actions.
4. Train until loss converges (warm start — does not need to be perfect).
5. Save the BC-trained policy head checkpoint to `artifacts/`.

**Done when:** BC loss converges and checkpoint is saved.

---

### Phase 9 — PPO Fine-tuning
**Module:** `doom_agent/training/ppo_trainer.py`

1. Wrap the environment as a Stable Baselines3 compatible `gym.Env`.
2. Define reward:
   - Dense: proportional progress toward active sub-goal
   - Sparse: fixed bonus on sub-goal completion
3. Load the BC-trained weights as the starting policy.
4. Run PPO training via Stable Baselines3.
5. Log rewards per episode to `artifacts/`.

**Done when:** PPO runs without error, reward improves over training, checkpoints save.

---

### Phase 10 — Evaluation
**Modules:** `doom_agent/agent/agent.py` + evaluation scripts

1. Load trained policy. Run on fixed training maps. Record sub-goal completion rate and episode efficiency.
2. Coordinate with Daniel: export trained weights so the same policy can be loaded in the Unity environment (via DoomLoader).
3. Run the same agent in Unity with no retraining. Record the same metrics.
4. Daniel retrains with Unity ML-Agents on the same architecture. Record metrics.
5. Produce the three-way comparison table: ViZDoom baseline vs. Unity (no retrain) vs. Unity (retrained).

**Done when:** All three metric sets are recorded and comparable.
