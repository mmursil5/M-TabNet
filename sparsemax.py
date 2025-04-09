import torch
import torch.nn as nn

class Sparsemax(nn.Module):
    def forward(self, input):
        original_size = input.size()
        input = input.view(input.size(0), -1)

        dim = 1
        number_of_logits = input.size(dim)

        input_sorted, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_sorted.cumsum(dim) - 1

        range_values = torch.arange(1, number_of_logits + 1, device=input.device).float()
        is_gt = (input_sorted - input_cumsum / range_values) > 0

        k = is_gt.float().sum(dim).unsqueeze(dim)
        tau_sum = input_cumsum.gather(dim, (k - 1).long())
        tau = tau_sum / k.float()

        output = torch.clamp(input - tau, min=0)
        return output.view(original_size)
