import torch
import torch.nn as nn
from torch.optim import Adam

class ActivePulseDistiller:
    """
    Implements Real-Time Supervised Fine-Tuning (SFT).
    Updates local persona masks when dissonance exceeds threshold.
    """
    def __init__(self, model, lr=1e-5):
        self.model = model
        # Only optimize the persona masks, not the whole backbone
        self.optimizer = Adam([model.vnm.masks], lr=lr)
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def step(self, hidden_states, teacher_logits):
        """
        Performs a 'Pulse' update. 
        Aligns the student's persona resonance with teacher's output.
        """
        self.optimizer.zero_grad()
        
        # Current persona activation
        current_output = self.model(hidden_states)
        
        # Distillation loss
        loss = self.loss_fn(
            torch.log_softmax(current_output, dim=-1),
            torch.softmax(teacher_logits, dim=-1)
        )
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
