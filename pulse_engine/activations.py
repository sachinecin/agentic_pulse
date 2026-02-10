import torch
import torch.nn as nn

class VirtualNeuronMask(nn.Module):
    """
    Implements Virtual Neuron Masking (VNM) to partition 
    hidden states into specialized personas during a forward pass.
    """
    def __init__(self, hidden_dim, persona_labels):
        super().__init__()
        self.persona_labels = persona_labels
        self.num_personas = len(persona_labels)
        # Learnable masks for each persona
        self.masks = nn.Parameter(torch.randn(self.num_personas, hidden_dim))
        
    def forward(self, hidden_states):
        # Apply masks to the hidden states to create persona-specific vectors
        # Shape: [NumPersonas, Batch, SeqLen, HiddenDim]
        mask_weights = torch.sigmoid(self.masks)
        return hidden_states.unsqueeze(0) * mask_weights.unsqueeze(1).unsqueeze(2)
