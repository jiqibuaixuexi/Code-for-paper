import paddle
import paddle.nn
from paddle.nn import functional as F
import numpy as np

METHODS = ['U-Ignore', 'U-Zeros', 'U-Ones', 'U-SelfTrained', 'U-MultiClass']
CLASS_NUM = [2015, 3374]
CLASS_WEIGHT = paddle.to_tensor([5389/i for i in CLASS_NUM])

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    # assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, axis=1)
    target_softmax = F.softmax(target_logits, axis=1)

    mse_loss = (input_softmax-target_softmax)**2 * CLASS_WEIGHT
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, axis=1)
    target_softmax = F.softmax(target_logits, axis=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div

class cross_entropy_loss(object):
    """
    map all uncertainty values to a unique value "2"
    """
    
    def __init__(self):
        self.base_loss = paddle.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, reduction='mean')
    
    def __call__(self, output, target):
        # target[target == -1] = 2
        output_softmax = F.softmax(output, axis=1) 
        
        return self.base_loss(output_softmax, target)

def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    # assert activations.size() == ema_activations.size()

    activations = paddle.reshape(activations, (activations.shape[0], -1))
    ema_activations = paddle.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = paddle.reshape(paddle.norm(similarity, 2, 1), (-1, 1)) 
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = paddle.reshape(paddle.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss
    