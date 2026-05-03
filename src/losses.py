import torch.nn.functional as F

def kl_divergence_loss(q, p, eps=1e-8):
    """
    KL(p || q): how much q diverges from the true distribution p.
    q = predicted distribution (model output, already softmaxed)
    p = true soft label distribution
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()

def js_divergence_loss(q, p, eps=1e-8):
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    m = 0.5 * (p + q)

    return 0.5 * (p * (p.log() - m.log())).sum(dim=-1).mean() + \
           0.5 * (q * (q.log() - m.log())).sum(dim=-1).mean()


def entropy_calibrated_kl_loss(q, p, alpha=0.5, eps=1e-8):
    """
    Custom loss = KL divergence + penalty for getting entropy wrong.

    Intuition: Beyond matching the distribution shape, we want the model
    to correctly predict HOW MUCH humans disagree (entropy level).
    A model that predicts the right top class but completely wrong
    entropy profile should be penalized extra.

    alpha controls the trade-off between distribution matching
    and entropy calibration.
    """
    kl = kl_divergence_loss(q, p, eps)

    true_entropy = -(p * (p + eps).log()).sum(dim=-1)
    pred_entropy = -(q * (q + eps).log()).sum(dim=-1)

    entropy_error = F.mse_loss(pred_entropy, true_entropy)

    return kl + alpha * entropy_error
