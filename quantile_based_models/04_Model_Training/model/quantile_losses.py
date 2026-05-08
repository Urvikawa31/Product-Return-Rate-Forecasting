import torch
import torch.nn as nn

class PinballLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super(PinballLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert preds.shape[1] == len(self.quantiles)
        loss = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            loss.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.cat(loss, dim=1)
        return torch.mean(torch.sum(loss, dim=1))

class TubeLoss(nn.Module):
    """
    Simultaneous upper/lower bound loss as defined in 2412.06853v3
    rho_t^r(y, mu1, mu2) where mu1 is lower, mu2 is upper.
    """
    def __init__(self, t=0.05, r=0.5):
        super(TubeLoss, self).__init__()
        self.t = t # Confidence level (alpha/2)
        self.r = r # Balancing factor

    def forward(self, preds, target):
        # preds: [batch, 2] -> mu1 (lower), mu2 (upper)
        mu1 = preds[:, 0]
        mu2 = preds[:, 1]
        y = target.squeeze()
        
        # Case 1: y > mu2
        c1 = self.t * (y - mu2)
        # Case 4: y < mu1
        c4 = self.t * (mu1 - y)
        
        # Case 2 & 3: mu1 <= y <= mu2
        # Transition point: r*mu2 + (1-r)*mu1
        transition = self.r * mu2 + (1 - self.r) * mu1
        c2 = (1 - self.t) * (mu2 - y)
        c3 = (1 - self.t) * (y - mu1)
        
        # Binary masks
        m1 = (y > mu2).float()
        m4 = (y < mu1).float()
        m_in = (1 - m1 - m4)
        m2 = m_in * (y >= transition).float()
        m3 = m_in * (y < transition).float()
        
        loss = m1 * c1 + m2 * c2 + m3 * c3 + m4 * c4
        return torch.mean(loss)
