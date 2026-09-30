import time

import genesis as gs
import torch

from src.sim import RukaSim


def main():
    gs.init()

    ruka = RukaSim(num_envs=1)

    while True:
        ruka.step(
            torch.zeros(
                1,
            )
        )
        time.sleep(1.0 / 60)
