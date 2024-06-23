import torch
import numpy as np

from ..attack import Attack


class JSMA(Attack):
    r"""
    Jacobian Saliency Map Attack in the paper 'The Limitations of Deep Learning in Adversarial Settings'
    [https://arxiv.org/abs/1511.07528v1]

    Distance Measure : L0

    Arguments:
        model (nn.Module): model to attack.
        theta (float): perturb length, range is either [theta, 0], [0, theta]. (Default: 1.0)
        gamma (float): highest percentage of pixels can be modified. (Default: 0.1)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.JSMA(model, theta=1.0, gamma=0.1)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, theta=1.0, gamma=0.1):
        super().__init__("JSMA", model)
        self.theta = theta
        self.gamma = gamma
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)
        else:
            # Because the JSMA algorithm does not use any loss function,
            # it cannot perform untargeted attacks indeed
            # (we have no control over the convergence of the attack to a data point that is NOT equal to the original class),
            # so we make the default setting of the target label is right circular shift
            # to make attack work if user didn't set target label.
            class_num = self.get_logits(torch.unsqueeze(images[0], 0)).shape[1]
            target_labels = (labels + 1) % class_num

        adv_images = images
        batch_size = images.shape[0]
        dim_x = int(np.prod(images.shape[1:]))
        max_iter = int(dim_x * self.gamma / 2)
        search_space = torch.ones(batch_size, dim_x).to(self.device)
        adv_prediction = torch.argmax(self.get_logits(adv_images), 1)

        # Algorithm 2
        i = 0
        while torch.sum(adv_prediction != target_labels) != 0 and i < max_iter and torch.sum(search_space != 0) != 0:
            grads_target, grads_other = self.compute_forward_derivative(
                adv_images, target_labels, class_num)

            p1, p2, valid = self.saliency_map(
                search_space, grads_target, grads_other, target_labels)

            cond = (adv_prediction != labels) & valid
            self.update_search_space(search_space, p1, p2, cond)

            xadv = self.modify_xadv(adv_images, batch_size, cond, p1, p2)
            adv_prediction = torch.argmax(self.get_logits(xadv))
            i += 1

        adv_images = torch.clamp(adv_images, min=0, max=1)
        return adv_images

    def jacobian(self, adv_images, c):
        tmp_images = adv_images.detach().clone().requires_grad_()
        output = self.get_logits(tmp_images)
        torch.sum(output[:, c]).backward()
        return tmp_images.grad.detach().clone()

    def compute_forward_derivative(self, adv_images, target_labels, class_num):
        jacobians = []
        for c in range(class_num):
            j = self.jacobian(adv_images, c)
            jacobians.append(j)

        jacobians = torch.stack(jacobians, 0)
        grads = jacobians.view((jacobians.shape[0], jacobians.shape[1], -1))
        grads_target = grads[target_labels, range(len(target_labels)), :]
        grads_other = grads.sum(dim=0) - grads_target
        return grads_target, grads_other

    def sum_pair(self, grads, dim_x):
        return grads.view(-1, dim_x, 1) + grads.view(-1, 1, dim_x)

    def and_pair(self, cond, dim_x):
        return cond.view(-1, dim_x, 1) & cond.view(-1, 1, dim_x)

    def saliency_map(self, search_space, grads_target, grads_other, y):

        dim_x = search_space.shape[1]

        # alpha in Algorithm 3 line 2
        gradsum_target = self.sum_pair(grads_target, dim_x)
        # alpha in Algorithm 3 line 3
        gradsum_other = self.sum_pair(grads_other, dim_x)

        if self.theta > 0:
            scores_mask = (torch.gt(gradsum_target, 0) &
                           torch.lt(gradsum_other, 0))
        else:
            scores_mask = (torch.lt(gradsum_target, 0) &
                           torch.gt(gradsum_other, 0))

        scores_mask &= self.and_pair(search_space.ne(0), dim_x)
        scores_mask[:, range(dim_x), range(dim_x)] = 0

        valid = torch.ones(scores_mask.shape[0]).to(torch.bool).to(self.device)

        scores = scores_mask.float() * (-gradsum_target * gradsum_other)
        best = torch.max(scores.view(-1, dim_x * dim_x), 1)[1]
        p1 = torch.remainder(best, dim_x)
        p2 = (best / dim_x).long()
        return p1, p2, valid

    def modify_xadv(self, xadv, batch_size, cond, p1, p2):
        ori_shape = xadv.shape
        xadv = xadv.view(batch_size, -1)
        for idx in range(batch_size):
            if cond[idx] != 0:
                xadv[idx, p1[idx]] += self.theta
                xadv[idx, p2[idx]] += self.theta
        xadv = torch.clamp(xadv, min=0, max=1)
        xadv = xadv.view(ori_shape)
        return xadv

    def update_search_space(self, search_space, p1, p2, cond):
        for idx in range(len(cond)):
            if cond[idx] != 0:
                search_space[idx, p1[idx]] = 0
                search_space[idx, p2[idx]] = 0
