"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Sliding window buffer for timestep vectors.
buffer.py
"""

from collections import deque
import torch


class Buffer():
    """
    Sliding window buffer for timestep vectors.
    """
    def __init__(self, window_size=8, timestep_dim=1027):
        self.window_size = window_size
        self.timestep_dim = timestep_dim
        self.buffer = None

    def init(self):
        print("Buffer!")
        self.buffer = deque(maxlen=self.window_size)

    def reset(self):
        self.buffer.clear()

    def _build_hud_vec(self, health, ammo, armor):
        '''
        DOOM caps all three at 200 (Megasphere / blue armor / bullet limit)
        We divide by 200 to normalize this data [0, 1].  CLIP embeddings
        re unit-length vectors (roughly +-0.1).  If the HUD information was
        not normalized, the transformer attention would be dominated by this
        information due to scale mismatch and drown out the visual signal.
        '''
        return torch.tensor([
            health / 200.0,
            ammo   / 200.0,
            armor  / 200.0,
        ], dtype=torch.float32)

    def push(self, frame_emb, goal_emb, health, ammo, armor):
        '''
        Step 1. Pack one frame's data into a single [1027] timestep vector.
        To build the timespec vector we join 1D tensors end-to-end along
        their axis.  The result is one flat vector representing:
        "what I see + what I'm trying to do + how I'm doing right now.
        ----------------
        frame_emb [512]
        goal_emb  [512]
        hud_vec   [  3]
        ----------------
        timestep  [1027]
        '''
        hud_vec = self._build_hud_vec(health, ammo, armor)
        timestep_vec = torch.cat([frame_emb, goal_emb, hud_vec], dim=0) # [1027]
        self.buffer.append(timestep_vec)

    def get_window(self):
        '''
        Step 2. Maintain a sliding window of the last 8 timestep vectors.
        Returns a [8, 1027] tensor the transformer will read.

        A sliding window always keeps the most recent N timestep vectors
        and drops the oldest when a new one arrives.  To do this we use
        the collections.deque tool with maxlen=N.  deque will drop the
        oldest queued value in the buffer so we don't have to manually
        track the index.

        A sliding window is used instead of the full episode history.
        This is because transformers have quadractic attention cost with
        sequence length.  Twice as long means four times the compute.
        A window of 8 gives enough temporal context for navigation decisions
        at a reasonable cost.  Larger sizes should be experimented with to
        assess impacts on observational stacking.
        '''
        n_real = len(self.buffer)
        n_pad = self.window_size - n_real

        padding = [torch.zeros(self.timestep_dim)] * n_pad
        window = padding + list(self.buffer)

        return torch.stack(window, dim=0) # [N, 1027]
