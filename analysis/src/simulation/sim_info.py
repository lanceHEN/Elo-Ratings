from dataclasses import dataclass, field

import numpy as np

@dataclass
class SimInfo:
    """Stores game/season information in one organized object for
    use in simulating games."""

    home_win_prob: float
    home_transition_matrices: np.ndarray
    away_transition_matrices: np.ndarray
