"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

Maps transformer output to actions.
policy_head.py
"""

import torch
import torch.nn as nn


class Policy_Head(nn.Module):
    """
    Maps transformer output to actions.  Outputs a single 256 vector.
    A compressed summary of "What I see and what I have seen."
    The policy head's job is to turn this into a decision:
    Which action should I take right now?
    """
    def __init__(self, hidden_dim=256, num_actions=8):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_actions)

    def forward(self, hidden_state):
        '''
        The linear layer outputs raw scores called logits.  Logits are
        any real number, positive or negative.  We keep logits separate
        from probabilities for behavior cloning as (bc) has cross-entropy
        loss.  Which internally calls log(softmax(logits)).  If we apply
        softmax first, and pass the probabilities PyTorch will have to undo
        it causing numerical instability.  This is why logits are kept for
        training and we only apply softmax when we need to sample or display
        probabilities.
        '''
        # hidden_state: [hidden_dim]
        return self.linear(hidden_state) # [num_actions] logits

    def select_action(self, logits, greedy=False):
        '''
        We apply softmax to our logits and then choose between sampling
        and greedy strategies.

        torch.multinomial samples randomly weighted by probability.  An action
        with 40% probability gets picked 40% of the time.  This is good for
        training and gives the agent exploration.

        torch.argmax always picks the highest probability action which is
        deterministic.  This is good for evaluation and see what the agent
        actually learns.
        '''
        probs = torch.softmax(logits, dim=-1)
        if greedy:
            return torch.argmax(probs).item()
        return torch.multinomial(probs, num_samples=1).item()
