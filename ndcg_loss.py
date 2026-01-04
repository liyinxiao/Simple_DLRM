#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""
This module contains implementation for Lambda NDCG loss (https://fburl.com/xzuax6kz).
Copied from DPER (https://fburl.com/code/k0lir44s)
"""

from typing import Optional, Tuple

import torch
from torch.autograd.function import FunctionCtx


class LambdaRankNdcgLoss(torch.nn.Module):
    """
    Compute LambdaRankNdcg loss (paper: https://fburl.com/an6r8wl2) given logit(before sigmoid), labels, and session_id.

    Call Args:
        logit(torch.Tensor): Predictions for each example
        label(torch.Tensor): Label for each example
        session_id(torch.Tensor): Session ID (scalar) for each example
        weight(Optional[torch.Tensor]): Optional weight for each example

    Example:
        >>> import random
        >>> n = 100
        >>> label = torch.randint(0, 4, (n,), dtype=torch.float32)
        >>> logit = torch.rand((n,), dtype=torch.float32)
        >>> session_ids =  torch.stack([torch.tensor(x) for x in sorted(random.choices([1, 2, 3], k=n))])
        >>> weight = torch.randint(0, 3, (n,), dtype=torch.float32)
        >>> ndcg_loss = LambdaRankNdcgLoss(use_weighted_loss=True)
        >>> result = ndcg_loss(logit, label, session_ids, weight)
    """

    def __init__(
        self,
        use_ndcg_as_loss: bool = True,
        use_exp_gain: bool = True,
        use_idcg_normalization: bool = True,
        use_weighted_loss: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.use_ndcg_as_loss = use_ndcg_as_loss
        self.use_exp_gain = use_exp_gain
        self.use_idcg_normalization = use_idcg_normalization
        self.use_weighted_loss = use_weighted_loss
        self.reduction = reduction
        # TODO(jiekie): temploportary block for `none` reduction
        if reduction == "none":
            raise AssertionError(
                "reduction=`none` is considered problematic, please use `mean` or `sum` instead"
            )

    def forward(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
        session_ids: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logit = torch.flatten(logit)
        label = torch.flatten(label)
        session_lengths = session_ids_to_lengths(session_ids).to(logit.device)
        loss, grad_loss = LambdaNdcgFunction.apply(
            logit,
            label,
            session_lengths,
            self.use_exp_gain,
            self.use_ndcg_as_loss,
            self.use_idcg_normalization,
        )

        if self.use_weighted_loss:
            weight = torch.flatten(
                    weight,
            )

            max_session_weight = torch.segment_reduce(
                data=weight,
                lengths=session_lengths,
                reduce="max",
                axis=0,
                unsafe=True,
                initial=0,
            )
            weight = max_session_weight.detach()
            loss = loss * weight

        if self.reduction == "mean":
            average_loss = torch.mean(loss)
            return average_loss
        elif self.reduction == "sum":
            return torch.sum(loss)
        elif self.reduction == "none":
            loss_per_example = torch.zeros_like(logit)
            for i, session_length in enumerate(session_lengths):
                loss_per_example[i : i + session_length] = loss[i]
            return loss_per_example
        else:
            raise NotImplementedError(f"reduction {self.reduction} is not supported")


class LambdaNdcgFunction(torch.autograd.Function):
    """
    Compute lambda ndcg loss using Pytorch.

    Note: This function requires examples with same session are grouped together, like:
    Examples: [E0, E1, E2, E3, E4, E5]
    Sessions: [A,  A,  A,  B,  B,  C]

    The `session_lengths` needs a padding 0, like:
    session_lengths: [0, 3, 2, 1]

    Call Args:
        prediction(torch.Tensor): A tensor of predicted scores
        label(torch.Tensor): A tensor of labels
        session_lengths(torch.Tensor): A tensor of session lengths (padding 0) converted from session_ids
        use_exp_gain(bool): Whether to use exponential gain or not
        use_ndcg_as_loss(bool): Whether to use ndcg as loss or not
        use_idcg_normalization(bool): Whether to normalize the loss by idcg

    Returns:
        loss(torch.Tensor): A tensor of average approximate ndcg loss
        grad_loss(torch.Tensor): A tensor of approximate ndcg gradients
    """

    @staticmethod
    # pyre-ignore[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx: FunctionCtx,
        prediction: torch.Tensor,
        label: torch.Tensor,
        session_lengths: torch.Tensor,
        use_exp_gain: bool,
        use_ndcg_as_loss: bool,
        use_idcg_normalization: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, grad_loss = compute_lambda_ndcg(
            prediction,
            label,
            session_lengths,
            use_exp_gain,
            use_ndcg_as_loss,
            use_idcg_normalization,
        )
        ctx.save_for_backward(grad_loss, session_lengths[1:])
        return loss, grad_loss

    @staticmethod
    # pyre-ignore[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(
        ctx: FunctionCtx, loss_output: torch.Tensor, grad_output: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        (grad_loss, session_lengths) = ctx.saved_tensors  # pyre-ignore

        # pyre-ignore[6]
        grad_out = torch.zeros(torch.sum(session_lengths).item()).to(
            session_lengths.device
        )

        count = 0
        for i in range(len(session_lengths)):
            grad_out[count : count + session_lengths[i]] = (
                loss_output[i + 1] * grad_loss[count : count + session_lengths[i]]
            )
            count += session_lengths[i]

        return grad_out, None, None, None, None, None


@torch.fx.wrap
def compute_lambda_ndcg(
    prediction: torch.Tensor,
    label: torch.Tensor,
    session_lengths: torch.Tensor,
    use_exp_gain: bool,
    use_ndcg_as_loss: bool,
    use_idcg_normalization: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the average lambda NDCG loss from a group of sessions.

    Args:
        prediction(torch.Tensor): A tensor of predicted scores
        label(torch.Tensor): A tensor of labels
        session_lengths(torch.Tensor): A tensor of session lengths (padding 0) converted from session_ids
        use_exp_gain(bool): Whether to use exponential gain or not
        use_ndcg_as_loss(bool): Whether to use ndcg as loss or not
        use_idcg_normalization(bool): Whether to normalize the loss by idcg

    Returns:
        loss(torch.Tensor): A tensor of average approximate ndcg loss
        grad_loss(torch.Tensor): A tensor of approximate ndcg gradients
    """

    loss = torch.zeros_like(session_lengths, dtype=torch.float)
    grad_loss = torch.zeros_like(prediction, dtype=torch.float)

    cur_index = int(0)
    for i, session_length in enumerate(session_lengths):
        if session_length == 0:
            continue

        data_indexes = torch.arange(
            cur_index,
            cur_index + int(session_length),
            dtype=torch.long,
            device=prediction.device,
        )
        # TODO(wilsonhong): try if torch.narrow is better
        session_loss, grad_session_loss = compute_lambda_ndcg_by_session(
            prediction=torch.take(prediction, data_indexes),
            label=torch.take(label, data_indexes),
            use_exp_gain=use_exp_gain,
            use_ndcg_as_loss=use_ndcg_as_loss,
            use_idcg_normalization=use_idcg_normalization,
        )
        loss[i] = session_loss
        grad_loss[cur_index : cur_index + int(session_length)] = grad_session_loss
        cur_index += session_length

    return loss, grad_loss


@torch.fx.wrap
def compute_lambda_ndcg_by_session(
    prediction: torch.Tensor,
    label: torch.Tensor,
    use_exp_gain: bool,
    use_ndcg_as_loss: bool,
    use_idcg_normalization: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the lambda NDCG loss for one session.

    Args:
        prediction(torch.Tensor): A tensor of predicted scores
        label(torch.Tensor): A tensor of labels
        use_exp_gain(bool): Whether to use exponential gain or not
        use_ndcg_as_loss(bool): Whether to compute ndcg as loss or not
        use_idcg_normalization(bool): Whether to normalize idcg to 1 when computing loss

    Returns:
        loss(torch.Tensor): A tensor of approximate ndcg loss
    """
    grad_loss = torch.zeros_like(prediction)
    gain = torch.exp2(label) if use_exp_gain or not use_ndcg_as_loss else label
    prediction_discounts = get_position_discounts(prediction)
    label_discounts = get_position_discounts(label)
    gain = gain.type(torch.float32)

    idcg = gain @ label_discounts
    # Note that label is assumed to be always non-negative.
    idcg = torch.max(idcg, torch.tensor(1e-5))

    dcg = gain @ prediction_discounts

    if use_idcg_normalization:
        session_weight = idcg.item()
    else:
        session_weight = 1.0

    pair_diff_label_sign = torch.sign(get_pairwise_diff(label))
    pair_diff_label_x_prediction = pair_diff_label_sign * get_pairwise_diff(prediction)

    lambda_mat = torch.abs(
        torch.mul(
            get_pairwise_diff(prediction_discounts),
            get_pairwise_diff(gain),
        )
    )

    if use_ndcg_as_loss:
        loss = torch.sub(idcg, dcg).div(session_weight)
    else:
        loss = (
            torch.neg(
                torch.sum(
                    lambda_mat.mul(
                        torch.nn.functional.logsigmoid(pair_diff_label_x_prediction)
                    )
                )
            )
            / session_weight
        )

    grad_loss = (
        torch.sum(
            -lambda_mat
            * pair_diff_label_sign
            * torch.sigmoid(-pair_diff_label_x_prediction),
            1,
        )
        / session_weight
    )

    return loss, grad_loss


@torch.fx.wrap
def get_position_discounts(t: torch.Tensor) -> torch.Tensor:
    """
    Get position discounts for tensor for NDCG

    Args:
        t: Tensor for position discounts

    Returns:
        position_discount (torch.Tensor): A tensor for the position discount
    """
    orders = torch.argsort(torch.argsort(t, descending=True))
    return torch.reciprocal(torch.log2(orders.add(2.0))).type(torch.float32)


@torch.fx.wrap
def get_pairwise_diff(t: torch.Tensor) -> torch.Tensor:
    """
    Get pairwise diff of a tensor for NDCG

    Args:
        t: Tensor for pairwise diff

    Returns:
        pairwise_diff (torch.Tensor): A tensor for the pairwise diff
    """
    matrix = torch.unsqueeze(t, 1).type(torch.float32) @ torch.ones(
        1, len(t), device=t.device
    )
    return matrix - matrix.T


@torch.fx.wrap
def session_ids_to_lengths(session_ids: torch.Tensor) -> torch.Tensor:
    """
    Args:
        session_ids: Tensor of session ids (scalar) for each example.
    Returns:
        session_lengths: Tensor of session lengths (padding 0).

    TODO(wilsonhong): Removing padding 0 from the result will cause loss function generate different value.
    For now I will keep it to match DPER's implementation.

    Exameple:
        >>> a = torch.tensor(1)
        >>> b = torch.tensor(2)
        >>> c = torch.tensor(3)
        >>> session_ids = [a, a, b, b, b, c]
        >>> offsets = session_ids_to_offsets(session_ids)
        >>> print(offsets)
        [0, 2, 3, 1]
    """
    length = 1
    session_lengths_list = [0]

    session_ids_cpu = session_ids.to(device="cpu", non_blocking=True)
    for i in range(session_ids_cpu.size(0) - 1):
        if torch.equal(session_ids_cpu[i], session_ids_cpu[i + 1]):
            length += 1
        else:
            session_lengths_list.append(length)
            length = 1
    session_lengths_list.append(length)

    return torch.tensor(session_lengths_list)
