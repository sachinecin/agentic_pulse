import torch
import torch.nn as nn

class DifferentiableConsensus(nn.Module):
    """
    Calculates semantic dissonance between personas and 
    resolves conflicts via vector alignment.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, 1)

    def forward(self, persona_states):
        # Simple mean-based consensus for the boilerplate
        # Real-world: Use a micro-recurrent loop to align diverging vectors
        consensus_vector = torch.mean(persona_states, dim=0)
        return consensus_vector
