"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Behavior cloning training loop
bc_trainer.py
"""

from collections import Counter
from pathlib import Path
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from memory.buffer import Buffer
from model.policy_head import Policy_Head
from model.temporal_transformer import Temporal_Transformer


class BC_Trainer():
    """
    Behavior cloning training loop.

    Supervised learning over the relabeled dataset: given a sliding window
    of the last 8 timesteps and an instruction, predict the action the
    demonstrator took.  Cross-entropy between the policy head's logits and
    the recorded action index.  This gives the policy a warm start before
    PPO fine-tuning.
    """
    def __init__(self, config=None, encoder=None):
        self.config = config
        self.encoder = encoder
        self.device = self.config['device']
        self.artifacts_path = Path(self.config['artifacts_path'])
        self.relabeled_path = self.artifacts_path / "relabeled"
        self.embeddings_path = self.artifacts_path / "embeddings"
        self.checkpoints_path = self.artifacts_path / "checkpoints"
        self.transformer = None
        self.policy_head = None
        self.num_actions = None
        self.goal_cache = {}
        self.instructions = []

    def init(self):
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)

    def register_goal(self, instruction, goal_emb):
        '''
        Record an instruction and its embedding in the in-memory caches.

        This must run with fresh encoding and disk-cache hit.  The disk
        cache stores goal embeddings inside each segment but self.instructions /
        self.goal_cache are in-memory state that has to be rebuilt every run.
        '''
        if instruction not in self.goal_cache:
            self.goal_cache[instruction] = goal_emb
            self.instructions.append(instruction)

    def encode_goal(self, text):
        '''
        Encode an instruction string once and cache the embedding.
        '''
        if text not in self.goal_cache:
            self.register_goal(text, self.encoder.encode_subgoal(text))
        return self.goal_cache[text]

    def encode_file(self, path):
        '''
        Encode one relabeled episode into tensors, cached on disk.

        CLIP is frozen so a frame's embedding never changes.  Without
        caching, every epoch would re-run the frozen encoder over every
        frame and training time would be dominated by the part of the model
        that never learns.  Without the cache, CLIP runs exactly once per
        collected dataset and training touches only the transformer and
        policy head.

        The cache is invalidated when the relabeled file is newer than
        the cached tensors (mtime check) -- re-collecting or re-relabeling
        triggers re-encoding automatically.

        HUD values are cached raw (not normalized).  Normalization stays
        in the Buffer._build_hud_vec so there is exactly one path for it
        in training and inference.

        The cache stores a dict assuring the segment list is not empty.
        This is because num_actions must survive the cache.  On a fully
        cached run the relabeled .pkl is never opened so anything the
        trainer needs has to live in the cache file.
        '''
        cache = self.embeddings_path / (path.stem + ".pt")
        if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
            data = torch.load(cache)
            self.num_actions = data["num_actions"]
            for seg in data["segments"]:
                self.register_goal(seg["instruction"], seg["goal_emb"])
            return data["segments"]

        with open(path, "rb") as f:
            relabeled = pickle.load(f)
        self.num_actions = len(relabeled["actions_table"])

        encoded = []
        for seg in relabeled["segments"]:
            frame_embs = torch.stack(
                [self.encoder.encode_frame(s["frame"])
                 for s in seg["steps"]])                         # [T, 512]
            goal_emb = self.encode_goal(seg["instruction"])      # [512]
            hud = torch.tensor([s["hud"] for s in seg["steps"]],
                               dtype=torch.float32)              # [T, 3]
            actions = torch.tensor([s["action"] for s in seg["steps"]],
                                   dtype=torch.long)             # [T]
            encoded.append({
                "instruction": seg["instruction"],
                "frame_embs": frame_embs,
                "goal_emb": goal_emb,
                "hud": hud,
                "actions": actions,
            })
        data = {"num_actions": self.num_actions, "segments": encoded}
        torch.save(data, cache)
        print(f"Encoded {path.name}: {len(encoded)} segments")
        return encoded

    def build_windows(self, encoded_segments):
        '''
        Turn encoded segments into (window, action, instruction) training
        samples using the same Buffer class the live agent uses.  Sharing the
        path gurantees training windows and inference windows are
        built identically.  Identical concatentation order, and zero padding
        at segment starts.

        The target for each window is the action taken at the window's final
        timestep: "given everything up to now, what did the demonstrator do next?"
        '''
        buffer = Buffer()
        buffer.init()

        windows, actions, goal_ids = [], [], []
        for seg in encoded_segments:
            buffer.reset()
            goal_id = self.instructions.index(seg["instruction"])
            for t in range(len(seg["actions"])):
                health, ammo, armor = seg["hud"][t].tolist()
                buffer.push(seg["frame_embs"][t], seg["goal_emb"],
                            health, ammo, armor)
                windows.append(buffer.get_window())
                actions.append(seg["actions"][t])
                goal_ids.append(goal_id)

        return (torch.stack(windows),                     # [M, 8, 1027]
                torch.stack(actions),                     # [M]
                torch.tensor(goal_ids, dtype=torch.long)) # [M]

    def build_dataset(self):
        '''
        Split train/validation by episode file and not by window.

        Adjacent windows share 7 of their 8 timesteps.  A random window-level
        split would place near-duplicates of training windows into validation
        and report inflated accuracy (data leakage).  Holding out whole
        episodes keeps validation honest: the model is scored on gameplay
        it has never seen any part of.
        '''
        paths = sorted(self.relabeled_path.glob("*.pkl"))
        if not paths:
            raise FileNotFoundError(
                f"No relabeled data in {self.relabeled_path} "
                "run 'make relabeled data' first.")

        val_paths = paths[::10] # every 10th episode -> validation
        train_paths = [p for p in paths if p not in val_paths]

        train_segments, val_segments = [], []
        for path in train_paths:
            train_segments += self.encode_file(path)
        for path in val_paths:
            val_segments += self.encode_file(path)

        train = self.build_windows(train_segments)
        val = self.build_windows(val_segments)
        print(f"Dataset: {len(train[0])} train windows "
              f"({len(train_paths)} episodes), "
              f"{len(val[0])} val windows ({len(val_paths)} episodes)")
        return train, val

    def evaluate(self, X, y, goal_ids, batch_size=256):
        '''
        Validation accuracy, overall and per instruction.

        eval() turns dropout OFF.  In train mode the transformer randomly
        zeroes 10% of the activations for regularization which would make
        validation scores noisy and underrated.  no_grad() skips building
        the autograd graph since nothing is being updated.

        Per-instruction accuracy is the number that matters:
        high overall accuracy could just mean the model learned the
        dominant instruction ("fire the weapon" is ~35% of steps) and
        ignored the rest.
        '''
        self.transformer.eval()
        self.policy_head.eval()

        correct = Counter()
        total = Counter()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = X[i:i + batch_size].to(self.device)
                yb = y[i:i + batch_size]
                gb = goal_ids[i:i + batch_size]
                logits = self.policy_head.forward(
                    self.transformer.forward(xb))
                preds = logits.argmax(dim=-1).cpu()
                for pred, target, goal in zip(preds, yb, gb):
                    total[goal.item()] += 1
                    if pred == target:
                        correct[goal.item()] += 1

        overall = sum(correct.values()) / max(sum(total.values()), 1)
        per_goal = {self.instructions[g]: correct[g] / total[g]
                    for g in total}
        return overall, per_goal

    def save_checkpoint(self, epoch, val_acc):
        '''
        Save everything PPO fine-tuning needs to reconstruct the model:
        both state dicts plus the architecture metadata.
        '''
        checkpoint = {
            "transformer": self.transformer.state_dict(),
            "policy_head": self.policy_head.state_dict(),
            "num_actions": self.num_actions,
            "hidden_dim": self.transformer.hidden_dim,
            "instructions": self.instructions,
            "epoch": epoch,
            "val_acc": val_acc,
        }
        torch.save(checkpoint, self.checkpoints_path / "bc_policy.pt")

    def train(self, epochs=30, batch_size=64, lr=3e-4):
        '''
        AdamW jointly optimizes the transformer and policy head.  CLIP
        never appears in the optimizer and stays frozen.

        The checkpoint keeps the best validation accuracy seen so far
        (not the last epoch) so overfitting late in training cannot
        degrade the saved policy.
        '''
        torch.manual_seed(0) # reproducible runs

        (X_tr, y_tr, g_tr), (X_val, y_val, g_val) = self.build_dataset()
        loader = DataLoader(TensorDataset(X_tr, y_tr),
                            batch_size=batch_size, shuffle=True)

        self.transformer = Temporal_Transformer().to(self.device)
        self.policy_head = Policy_Head(
            hidden_dim=256, num_actions=self.num_actions).to(self.device)

        params = (list(self.transformer.parameters())
                  + list(self.policy_head.parameters()))
        optimizer = torch.optim.AdamW(params, lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(1, epochs+1):
            self.transformer.train()
            self.policy_head.train()

            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                logits = self.policy_head.forward(
                    self.transformer.forward(xb))
                loss = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)

            avg_loss = epoch_loss / len(X_tr)
            val_acc, per_goal = self.evaluate(X_val, y_val, g_val)
            marker = ""
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(epoch, val_acc)
                marker = " *saved*"
            print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | "
                  f"val acc {val_acc:.3f}{marker}")

        print(f"\nBest val accuracy: {best_acc:.3f} "
              f"(random baseline: {1.0 / self.num_actions:.3f})")
        print("Per-instruction val accuracy (final epoch):")
        for instruction, acc in per_goal.items():
            print(f"  {instruction:<20} {acc:.3f}")
