"""
Cauchy Closed-Loop Convergence Monitor for Iterative Cold Diffusion.

Monitors successive reconstruction updates:
    Δ_k = || x^{(k)} - x^{(k-1)} ||_2 / (|| x^{(k-1)} ||_2 + ε)

Terminates sampling early if:
1. Convergence threshold reached: Δ_k < tolerance (reconstruction stabilized).
2. Inflection / divergence detected: update norm increases for 2 consecutive steps (preventing over-smoothing).

Completely architecture-agnostic: Works for U-Net, DiT, and any custom diffusion backbone.
"""

from __future__ import annotations

import torch



class CauchyConvergenceMonitor:
    def __init__(self, tolerance: float = 0.008, patience: int = 1, min_steps: int = 2):
        """
        Args:
            tolerance: Relative update threshold below which sampling is considered converged.
            patience: Number of consecutive steps with increasing residual before halting.
            min_steps: Minimum number of steps to run before allowing early stopping.
        """
        self.tolerance = tolerance
        self.patience = patience
        self.min_steps = min_steps

        self.history: list[float] = []
        self.prev_x: torch.Tensor | None = None
        self.consecutive_increases = 0

    def reset(self):
        """Reset state for a new restoration run."""
        self.history.clear()
        self.prev_x = None
        self.consecutive_increases = 0

    def check(self, curr_x: torch.Tensor, step_idx: int, total_steps: int) -> tuple[bool, float, str]:
        """
        Checks convergence after completing step_idx (0-indexed).

        Args:
            curr_x: Current reconstructed tensor (B, C, H, W).
            step_idx: Index of completed step (0-indexed).
            total_steps: Maximum planned steps.

        Returns:
            should_stop: Boolean flag indicating whether to terminate early.
            delta: Float relative difference metric.
            status_msg: Human-readable explanation.
        """
        if self.prev_x is None:
            self.prev_x = curr_x.detach().clone()
            return False, 1.0, f"Step 1/{total_steps} (Baseline)"

        # Relative Cauchy residual norm
        diff_norm = torch.norm(curr_x - self.prev_x).item()
        base_norm = torch.norm(self.prev_x).item() + 1e-6
        delta = diff_norm / base_norm

        self.history.append(delta)
        self.prev_x = curr_x.detach().clone()

        current_step_num = step_idx + 1

        # Check inflection / divergence
        if len(self.history) >= 2:
            if self.history[-1] > self.history[-2]:
                self.consecutive_increases += 1
            else:
                self.consecutive_increases = 0

        # Enforce minimum steps
        if current_step_num < self.min_steps:
            return False, delta, f"Step {current_step_num}/{total_steps} (Δ={delta:.4f})"

        # Check 1: Convergence tolerance met
        if delta < self.tolerance:
            return True, delta, f"Converged at step {current_step_num}/{total_steps} (Δ={delta:.4f} < {self.tolerance})"

        # Check 2: Inflection / Over-diffusion detected
        if self.consecutive_increases >= self.patience:
            return True, delta, f"Early stop: Inflection detected at step {current_step_num}/{total_steps} (Δ={delta:.4f})"

        return False, delta, f"Step {current_step_num}/{total_steps} (Δ={delta:.4f})"
