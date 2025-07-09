import torch
import torch.nn as nn

# A regular linear layer from a pretrained model
# Example: input size = 16, output size = 32
linear_layer = nn.Linear(16, 32)

# We freeze the original weights
for param in linear_layer.parameters():
    param.requires_grad = False


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank):
        super().__init__()
        self.original_layer = original_layer

        # Get the dimensions of the original layer
        in_features  = original_layer.in_features
        out_features = original_layer.out_features

        # Create the two smaller matrices, A and B
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        # The original, frozen layer's output
        original_output = self.original_layer(x)

        # The LoRA update
        lora_update = torch.matmul(
                                    torch.matmul(
                                                    x, 
                                                    self.lora_A
                                                ), 
                                    self.lora_B
                                    )
        # Return the original output plus the LoRA update

        return original_output + lora_update

# Let's create our LoRA-adapted layer with rank=4
lora_layer = LoRALayer(linear_layer, rank=4)

# Print parameter counts for comparison
original_params = linear_layer.in_features * linear_layer.out_features
lora_a_params = linear_layer.in_features * 4  # rank = 4
lora_b_params = 4 * linear_layer.out_features  # rank = 4
total_lora_params = lora_a_params + lora_b_params

print(f"Original layer parameters: {original_params}")
print(f"LoRA matrix A parameters: {lora_a_params}")
print(f"LoRA matrix B parameters: {lora_b_params}")
print(f"Total LoRA parameters: {total_lora_params}")
print(f"Parameter reduction: {((original_params - total_lora_params) / original_params * 100):.1f}%")

# You would then train only the `lora_A` and `lora_B` parameters.