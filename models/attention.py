import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # This layer will process the concatenated encoder and decoder hidden states
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        # This layer will produce the final energy score for each encoder state
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden shape = [1, batch_size, hidden_dim]
        # encoder_outputs shape = [batch_size, src_len, hidden_dim]

        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        # --- START OF FIX ---

        # 1. Permute hidden to be [batch_size, 1, hidden_dim] so batch is first
        hidden = hidden.permute(1, 0, 2)

        # 2. Repeat the decoder hidden state src_len times to match encoder_outputs
        # We want to compare the single decoder state with EACH of the src_len encoder states
        hidden = hidden.repeat(1, src_len, 1)
        # Now, hidden shape = [batch_size, src_len, hidden_dim]
        # And encoder_outputs shape = [batch_size, src_len, hidden_dim]
        # The shapes now match perfectly for concatenation.

        # 3. Concatenate along the last dimension (the feature dimension)
        concat_energy = torch.cat((hidden, encoder_outputs), dim=2)
        # concat_energy shape = [batch_size, src_len, hidden_dim * 2]

        # 4. Pass through the attention layer
        energy = torch.tanh(self.attn(concat_energy))
        # energy shape = [batch_size, src_len, hidden_dim]
        
        # --- END OF FIX ---

        # 5. Get the final attention score (a single number per encoder state)
        attention = self.v(energy).squeeze(2)
        # attention shape = [batch_size, src_len]

        # 6. Apply softmax to get weights that sum to 1
        return F.softmax(attention, dim=1)