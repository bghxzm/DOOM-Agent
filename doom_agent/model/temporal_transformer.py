"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Transformer Encoder with causal masking.
temporal_transformer.py
"""

import math
import torch
import torch.nn as nn


class Temporal_Transformer(nn.Module):
    """
    Transformer Encoder with causal masking.
    """
    def __init__(self, input_dim=1027, hidden_dim=256, nhead=4,
                 num_layers=2, dim_feedforward=512, dropout=0.1, window_size=8):
        '''
        Step 1 - Input projection

        1027 is an awkward dimension so we project down to 256 with a
        single layer.  hidden_dim must be divisible by the number of
        attention heads.  Since nhead=4 each head gets a standard size of
        64 dimensions.

        register_buffer stores the positional encoding tensor as part of the
        module but not as a learnable parameter.  This means that:
        1. It moves automatically when you call .to('cuda') or .to('mps')
        2. It is included in state_dict() saves
        3. It has no gradient - sinusoidal pattern is fixed math (not trained)

        If we just stored pos_encoding the register buffer would stay on the
        CPU even after moving the model to GPU and we would get a device
        mismatch.
        '''
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim) # applied to each N row independently
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.register_buffer('pos_encoding', self._build_pos_encoding(window_size, hidden_dim))

    def _build_pos_encoding(self, max_len, d_model):
        '''
        Step 2 - Positional encoding

        Transformers have no sense of order and treat input as a set and not
        a sequence.  This is problematic for a temporal transformer because
        we are looking to conceptualize sequences to determine the next action.

        Positional encoding is used to fix this problem by making each timestep
        distinct.  A small unique signal is added to each row based on its
        position in the sequence.

        Each position gets a distinct pattern of sines and cosines across the
        feature dimensions.  The model then learns to detect these patterns and
        reason about distance between positions.

        For simplicity we use sinusoidal encoding (fixed math not learned parameters)
        '''
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe # [max_len, hidden_dim]

    def forward(self, x):
        '''
        Step 3 - Casual Masking

        Without masking, position 3 can attend to future positions 4,5,6,7.
        During training that would let the model cheat by looking ahead.  At
        inference time the future does not exist so it would fail.

        A causal mask sets future positions to -inf before the softmax which
        makes them contribute 0 to attention.  This creates a lower-triangular
        attention matrix:

        Position 0 can see: [0]
        Position 1 can see: [0, 1]
        Position 2 can see: [0, 1, 2]
        ...
        Position 7 can see: [0, 1, 2, 3, 4, 5, 6, 7]

        The policy head only needs one output vector per timestep.  We take
        the last one (index-1) because it has attended all previous positions
        and accumulated the full context of the window.  This represents:
        "given everything I've seen, what is the current state?"

        Accepts either a single window [N, 1027] (inference, returns
        [hidden_dim]) or a batch of windows [B, N, 1027] (training,
        returns [B, hidden_dim]).  The positional encoding [N, hidden_dim]
        broadcasts across the batch dimension and the causal mask is the
        same for every window in the batch.
        '''
        single = (x.dim() == 2)
        if single:
            x = x.unsqueeze(0)                # [1, N, 1027]

        x = self.input_proj(x)                # [1, N, 1027]
        x = x + self.pos_encoding[:x.size(1)]

        mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
        out = self.transformer(x, mask=mask, is_causal=True) # [B, N, hidden_dim]
        out = out[:, -1, :] # [B, hidden_dim]
        return out.squeeze(0) if single else out
