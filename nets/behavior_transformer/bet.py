"""
Note: This is copied from https://github.com/notmahi/bet, with a little bit of modification.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .gpt import GPT
from .utils import MLP

class KMeansDiscretizer:
    """
    Simplified and modified version of KMeans algorithm from sklearn.
    """

    def __init__(
        self,
        num_bins: int = 100,
        kmeans_iters: int = 50,
    ):
        super().__init__()
        self.n_bins = num_bins
        self.kmeans_iters = kmeans_iters

    def fit(self, input_actions: torch.Tensor) -> None:
        self.bin_centers = KMeansDiscretizer._kmeans(
            input_actions, ncluster=self.n_bins, niter=self.kmeans_iters
        )

    @classmethod
    def _kmeans(cls, x: torch.Tensor, ncluster: int = 512, niter: int = 50):
        """
        Simple k-means clustering algorithm adapted from Karpathy's minGPT libary
        https://github.com/karpathy/minGPT/blob/master/play_image.ipynb
        """
        N, D = x.size()
        c = x[torch.randperm(N)[:ncluster]]  # init clusters at random

        pbar = tqdm.trange(niter)
        pbar.set_description("K-means clustering")
        for i in pbar:
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            if ndead:
                tqdm.tqdm.write(
                    "done step %d/%d, re-initialized %d dead clusters"
                    % (i + 1, niter, ndead)
                )
            c[nanix] = x[torch.randperm(N)[:ndead]]  # re-init dead clusters
        return c


class BehaviorTransformer(nn.Module):
    GOAL_SPEC = Enum("GOAL_SPEC", "concat stack unconditional")

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        goal_dim: int,
        gpt_model: GPT,
        n_clusters: int = 32,
        kmeans_fit_steps: int = 500,
        kmeans_iters: int = 50,
        offset_loss_multiplier: float = 1.0e3,
        gamma: float = 2.0,
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._goal_dim = goal_dim

        if goal_dim <= 0:
            self._cbet_method = self.GOAL_SPEC.unconditional
        elif obs_dim == goal_dim:
            self._cbet_method = self.GOAL_SPEC.concat
        else:
            self._cbet_method = self.GOAL_SPEC.stack

        self._gpt_model = gpt_model
        # For now, we assume the number of clusters is given.
        # assert n_clusters > 0 and kmeans_fit_steps > 0
        assert n_clusters > 0
        self._K = n_clusters
        self._kmeans_fit_steps = kmeans_fit_steps
        self._clustering_algo = KMeansDiscretizer(
            num_bins=n_clusters, kmeans_iters=kmeans_iters
        )
        self._current_steps = 0
        self._map_to_cbet_preds = MLP(
            in_channels=gpt_model.config.output_dim,
            hidden_channels=[(act_dim + 1) * n_clusters],
        )
        self._collected_actions = []
        self._have_fit_kmeans = False
        self._offset_loss_multiplier = offset_loss_multiplier
        # Placeholder for the cluster centers.
        self.register_buffer("_cluster_centers", torch.zeros((n_clusters, act_dim)).float())
        self._criterion = FocalLoss(gamma=gamma)

    def forward(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._predict(obs_seq, goal_seq, action_seq)

    def _fit_kmeans_from_action_dataset(
        self,
        actions: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # actions: (N, act_dim)
        assert not self._have_fit_kmeans, "Have fitted kmeans already."
        self._cluster_centers = self._cluster_centers.to(actions.device)
        logging.info("Fitting KMeans")
        self._clustering_algo.fit(actions)
        self._have_fit_kmeans = True
        self._cluster_centers = self._clustering_algo.bin_centers.float().to(actions.device)

    def _predict(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        # Assume dimensions are N T D for N sequences of T timesteps with dimension D.
        if self._cbet_method == self.GOAL_SPEC.unconditional:
            gpt_input = obs_seq
        elif self._cbet_method == self.GOAL_SPEC.concat:
            gpt_input = torch.cat([goal_seq, obs_seq], dim=1)
        elif self._cbet_method == self.GOAL_SPEC.stack:
            gpt_input = torch.cat([goal_seq, obs_seq], dim=-1)
        else:
            raise NotImplementedError

        gpt_output = self._gpt_model(gpt_input)
        if self._cbet_method == self.GOAL_SPEC.concat:
            # Chop off the goal encodings.
            gpt_output = gpt_output[:, goal_seq.size(1) :, :]
        cbet_preds = self._map_to_cbet_preds(gpt_output)
        cbet_logits, cbet_offsets = torch.split(
            cbet_preds, [self._K, self._K * self._act_dim], dim=-1
        )
        cbet_offsets = einops.rearrange(cbet_offsets, "N T (K A) -> N T K A", K=self._K)

        cbet_probs = torch.softmax(cbet_logits, dim=-1)
        N, T, choices = cbet_probs.shape
        if action_seq is not None:
            # Sample from the multinomial distribution, one per row.
            sampled_centers = einops.rearrange(
                torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
                "(N T) 1 -> N T 1",
                N=N,
            )
        else:
            # Greedy selection, but emprically, this doesn't boost the performance.
            sampled_centers = einops.rearrange(
                torch.argmax(cbet_probs, dim=-1), "N T -> N T 1", N=N,
            )
        flattened_cbet_offsets = einops.rearrange(cbet_offsets, "N T K A -> (N T) K A")
        sampled_offsets = flattened_cbet_offsets[
            torch.arange(flattened_cbet_offsets.shape[0]), sampled_centers.flatten()
        ].view(N, T, self._act_dim)
        centers = self._cluster_centers[sampled_centers.flatten()].view(
            N, T, self._act_dim
        )
        if action_seq is not None:
            # Figure out the loss for the actions.
            # First, we need to find the closest cluster center for each action.
            action_bins = self._find_closest_cluster(action_seq)
            true_offsets = action_seq - self._cluster_centers[action_bins]
            predicted_offsets = flattened_cbet_offsets[
                torch.arange(flattened_cbet_offsets.shape[0]), action_bins.flatten()
            ].view(N, T, self._act_dim)
            # Now we can compute the loss.
            offset_loss = F.mse_loss(predicted_offsets, true_offsets)
            cbet_loss = self._criterion(  # F.cross_entropy
                einops.rearrange(cbet_logits, "N T D -> (N T) D"),
                einops.rearrange(action_bins, "N T -> (N T)"),
            )
            loss = cbet_loss + self._offset_loss_multiplier * offset_loss
            pred_action_bins = torch.argmax(cbet_logits[:, -1], dim=-1) # only the last action will be used
            gt_action_bins = action_bins[:, -1]
            cls_acc = (pred_action_bins == gt_action_bins).sum().item() / gt_action_bins.shape[0]
            loss_dict = {
                "classification_loss": cbet_loss.detach().cpu().item(),
                "offset_loss": offset_loss.detach().cpu().item(),
                "total_loss": loss.detach().cpu().item(),
                "cls_accuracy": cls_acc,
            }
            if self._current_steps < self._kmeans_fit_steps:
                loss *= 0.0
            return centers + sampled_offsets, loss, loss_dict
        return centers + sampled_offsets, None, {}

    def _find_closest_cluster(self, action_seq: torch.Tensor) -> torch.Tensor:
        N, T, _ = action_seq.shape
        flattened_actions = einops.rearrange(action_seq, "N T A -> (N T) A")
        cluster_center_distance = torch.sum(
            (flattened_actions[:, None, :] - self._cluster_centers[None, :, :]) ** 2,
            dim=2,
        )  # (N T) K A -> (N T) K
        closest_cluster_center = torch.argmin(cluster_center_distance, dim=1)  # (N T)
        discretized_action = einops.rearrange(
            closest_cluster_center, "(N T) -> N T", N=N, T=T
        )
        return discretized_action

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        optimizer = self._gpt_model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
        )
        optimizer.add_param_group({"params": self._map_to_cbet_preds.parameters()})
        return optimizer

    def save_model(self, path: Path):
        torch.save(self.state_dict(), path / "cbet_model.pt")
        torch.save(self._gpt_model.state_dict(), path / "gpt_model.pt")

    def load_model(self, path: Path):
        if (path / "cbet_model.pt").exists():
            self.load_state_dict(torch.load(path / "cbet_model.pt"))
        elif (path / "gpt_model.pt").exists():
            self._gpt_model.load_state_dict(torch.load(path / "gpt_model.pt"))
        else:
            logging.warning("No model found at %s", path)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
