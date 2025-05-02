import torch
import torch.nn as nn
import torch.nn.functional as F


class PGD_linf(nn.Module):
    def __init__(self, epsilon, num_steps, step_size):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach()

        # uniform PGD initialization
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        for _ in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx)
                # logits = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits, by, reduction="sum")
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())

            adv_bx = torch.min(
                torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon
            ).clamp(0, 1)

        return adv_bx
