import torch
import torch.nn as nn
import torch.optim as optim

from ..attack import Attack


class ZOO(Attack):
    r"""
    ZOO in the paper 'ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models'
    [https://arxiv.org/abs/1708.03999]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.ZOO(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, init_c=1, kappa=0, steps=50, lr=0.01, binary_search_steps=9, abort_early=True, adam_beta1=0.9, adam_beta2=0.999, reset_adam_after_found=False):
        super().__init__("ZOO", model)
        self.init_c = init_c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.abort_early = abort_early
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2
        self.reset_adam_after_found = reset_adam_after_found
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            labels = self.get_target_label(images, labels)

        # ADAM status
        self.mt = torch.zeros(images.shape).to(self.device)
        self.vt = torch.zeros(images.shape).to(self.device)
        self.adam_epoch = torch.ones(images.shape).to(self.device)
        self.real_modifier = torch.zeros(
            images.shape).unsqueeze(0).to(self.device)
        self.var_list = torch.arange(0, torch.prod(images)).to(self.device)
        self.hess = torch.ones(images.shape).to(self.device)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        o_best_adv_images = images.clone().detach()

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        batch_size = len(images)
        lower_bound = torch.zeros((batch_size, )).to(self.device)
        const = torch.full((batch_size, ), self.init_c,
                           dtype=torch.float).to(self.device)
        upper_bound = torch.full((batch_size, ), 1e10).to(self.device)

        o_best_score = torch.full(
            (batch_size, ), -1, dtype=torch.long).to(self.device)
        o_best_Lx = torch.full((batch_size, ), 1e10).to(self.device)

        for _ in range(self.binary_search_steps):
            best_score = torch.full(
                (batch_size, ), -1, dtype=torch.long).to(self.device)
            best_Lx = torch.full((batch_size, ), 1e10).to(self.device)
            prev_cost = 1e10

            # reset ADAM status
            self.mt = torch.zeros(images.shape).to(self.device)
            self.vt = torch.zeros(images.shape).to(self.device)
            self.adam_epoch = torch.ones(images.shape).to(self.device)

            for step in range(self.steps):

                var = self.real_modifier.unsqueeze(
                    0).expand(batch_size * 2 + 1, -1)
                var_size = self.real_modifier.size
                var_indice = torch.randint(
                    0, self.var_list.shape[0], (batch_size, ), dtype=torch.long)
                indice = self.var_list[var_indice]

                for i in range(batch_size):
                    var[i * 2 + 1, indice[i]] += 0.0001
                    var[i * 2 + 2, indice[i]] -= 0.0001

                # Get adversarial images
                adv_images = self.tanh_space(w)

                # Calculate loss
                current_Lx = MSELoss(Flatten(adv_images),
                                     Flatten(images)).sum(dim=1)

                Lx_loss = current_Lx.sum()

                outputs = self.get_logits(adv_images)
                if self.targeted:
                    # f_loss = self.f(outputs, target_labels).sum()
                    f_loss = self.f(outputs, labels)
                else:
                    # f_loss = self.f(outputs, labels).sum()
                    f_loss = self.f(outputs, labels)

                cost = Lx_loss + torch.sum(const * f_loss)

                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                # Update adversarial images
                pre = torch.argmax(outputs.detach(), 1)
                condition_1 = self.compare(pre, labels)
                condition_2 = (current_Lx < best_Lx)
                # Filter out images that get either correct predictions or non-decreasing loss,
                # i.e., only images that are both misclassified and loss-decreasing are left
                mask_1_2 = torch.logical_and(condition_1, condition_2)
                best_Lx[mask_1_2] = current_Lx[mask_1_2]
                best_score[mask_1_2] = pre[mask_1_2]

                condition_3 = (current_Lx < o_best_Lx)
                o_mask = torch.logical_and(condition_1, condition_3)
                o_best_Lx[o_mask] = current_Lx[o_mask]
                o_best_score[o_mask] = pre[o_mask]

                o_best_adv_images[o_mask] = adv_images[o_mask]

                # Check if we should abort search if we're getting nowhere.
                if self.abort_early and step % (self.steps // 10) == 0:
                    if cost > prev_cost * 0.9999:
                        break
                    else:
                        prev_cost = cost

            # Adjust the constant as needed
            outputs = self.get_logits(adv_images)
            pre = torch.argmax(outputs, 1)

            condition_1 = self.compare(pre, labels)
            condition_2 = (best_score != -1)
            condition_3 = upper_bound < 1e9

            mask_1_2 = torch.logical_and(condition_1, condition_2)
            mask_1_2_3 = torch.logical_and(mask_1_2, condition_3)
            const_1 = (lower_bound + upper_bound) / 2.0

            upper_bound_min = torch.min(upper_bound, const)
            upper_bound[mask_1_2] = upper_bound_min[mask_1_2]
            const[mask_1_2_3] = const_1[mask_1_2_3]

            mask_n1_n2_3 = torch.logical_and(~mask_1_2, condition_3)
            upper_bound_max = torch.max(lower_bound, const)
            upper_bound[~mask_1_2] = upper_bound_max[~mask_1_2]
            const[mask_n1_n2_3] *= 10

        # print(o_best_Lx)
        return o_best_adv_images

    def compare(self, predition, labels):
        if self.targeted:
            # We want to let pre == target_labels in a targeted attack
            ret = (predition == labels)
        else:
            # If the attack is not targeted we simply make these two values unequal
            ret = (predition != labels)
        return ret

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # get the target class's logit
        real = torch.sum(one_hot_labels * outputs, dim=1)
        # find the max logit other than the target classs
        other = torch.max((1 - one_hot_labels) * outputs - one_hot_labels * 1e12, dim=1)[0]  # nopep8

        if self.targeted:
            return torch.clamp((other - real), min=-self.kappa)
        else:
            return torch.clamp((real - other), min=-self.kappa)

    def coordinate_ADAM(losses, indice, grad, hess, batch_size, mt_arr, vt_arr, real_modifier, up, down, lr, adam_epoch, beta1, beta2, proj):
        for i in range(batch_size):
            grad[i] = (losses[i*2+1] - losses[i*2+2]) / 0.0002 
        # true_grads = self.sess.run(self.grad_op, feed_dict={self.modifier: self.real_modifier})
        # true_grads, losses, l2s, scores, nimgs = self.sess.run([self.grad_op, self.loss, self.l2dist, self.output, self.newimg], feed_dict={self.modifier: self.real_modifier})
        # grad = true_grads[0].reshape(-1)[indice]
        # print(grad, true_grads[0].reshape(-1)[indice])
        # self.real_modifier.reshape(-1)[indice] -= self.LEARNING_RATE * grad
        # self.real_modifier -= self.LEARNING_RATE * true_grads[0]
        # ADAM update
        mt = mt_arr[indice]
        mt = beta1 * mt + (1 - beta1) * grad
        mt_arr[indice] = mt
        vt = vt_arr[indice]
        vt = beta2 * vt + (1 - beta2) * (grad * grad)
        vt_arr[indice] = vt
        # epoch is an array; for each index we can have a different epoch number
        epoch = adam_epoch[indice]
        corr = (np.sqrt(1 - np.power(beta2,epoch))) / (1 - np.power(beta1, epoch))
        m = real_modifier.reshape(-1)
        old_val = m[indice] 
        old_val -= lr * corr * mt / (np.sqrt(vt) + 1e-8)
        # set it back to [-0.5, +0.5] region
        if proj:
            old_val = np.maximum(np.minimum(old_val, up[indice]), down[indice])
        # print(grad)
        # print(old_val - m[indice])
        m[indice] = old_val
        adam_epoch[indice] = epoch + 1
