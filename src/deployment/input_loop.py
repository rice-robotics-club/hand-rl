"""Transport-independent sensor/goal sampling on the training control clock."""

import math
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from threading import Event
from typing import Protocol


class InputSource(Protocol):
    """An adapter for hardware APIs or subscriber buffers.

    Methods should return promptly or use bounded hardware timeouts. A subscriber
    adapter must reject missing/stale messages rather than inventing sensor values.
    """

    def read_sensors(self) -> Mapping[str, object]:
        """Read current sensor values in their documented units and frames."""
        ...

    def read_goal(self, elapsed_seconds: float) -> Mapping[str, object]:
        """Read the current goal at elapsed time since this input loop started."""
        ...


@dataclass(frozen=True)
class InputFrame:
    tick: int
    """Scheduled control tick; gaps indicate skipped deadlines."""
    sampled_at: float
    """Monotonic timestamp at the start of acquisition, not a sensor hardware timestamp."""
    elapsed_seconds: float
    sensors: Mapping[str, object]
    goal: Mapping[str, object]


def training_control_period(config: Mapping) -> float:
    """Resolve the interval used by the two existing simulator implementations.

    Physics substeps do not change the policy rate. Do not guess a Genesis default
    when the training config omits its timestep: require it to be recorded.
    """
    system = config.get("system", {})
    if system.get("class_name") == "src.sim.ruka_sim:RukaSim":
        period = system.get("control_dt", 1 / 60)
    elif system.get("class_name") == "src.sim.hand_piano:HandPiano":
        period = system.get("genesis", {}).get("sim_options", {}).get("dt")
    else:
        raise ValueError(
            "Input timing requires a supported training system configuration"
        )
    if period is None:
        raise ValueError(
            "Record the training control timestep explicitly in the config"
        )
    if isinstance(period, bool) or not isinstance(period, (float, int)):
        raise TypeError("Training control timestep must be a number")
    if not math.isfinite(period) or period <= 0:
        raise ValueError("Training control timestep must be finite and positive")
    return float(period)


class InputLoop:
    """Best-effort fixed-rate acquisition with absolute monotonic deadlines.

    Slow readers/consumers skip missed ticks instead of generating catch-up bursts.
    No normalization, inference, or actuator commands occur in this loop.
    """

    def __init__(
        self,
        period: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], object] | None = None,
    ):
        if isinstance(period, bool) or not math.isfinite(period) or period <= 0:
            raise ValueError("Input period must be finite and positive")
        self.period = period
        self.rate_hz = 1 / period
        self._clock = clock
        self._wait = wait

    def run(
        self,
        source: InputSource,
        on_frame: Callable[[InputFrame], None],
        *,
        max_frames: int | None = None,
        stop_event: Event | None = None,
    ) -> int:
        """Read both inputs each tick and deliver an independent snapshot.

        First read is immediate. Goal time uses actual elapsed wall time, so an
        overrun does not slow the score. Source/consumer errors propagate and stop
        acquisition. Stopping interrupts waits but cannot interrupt a blocked driver.
        """
        if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
            raise ValueError("max_frames must be a positive integer")
        stop = stop_event if stop_event is not None else Event()
        wait = self._wait if self._wait is not None else stop.wait
        origin = self._clock()
        tick = count = 0
        while not stop.is_set() and (max_frames is None or count < max_frames):
            deadline = origin + tick * self.period
            now = self._clock()
            while now < deadline and not stop.is_set():
                wait(deadline - now)
                now = self._clock()
            if stop.is_set():
                break
            # If the OS woke us a whole period late, discard old scheduled ticks.
            tick = max(tick, math.floor((now - origin) / self.period))
            elapsed = now - origin
            sensors = source.read_sensors()
            if not isinstance(sensors, Mapping):
                raise TypeError("read_sensors() must return a mapping")
            sensors = deepcopy(dict(sensors))
            goal = source.read_goal(elapsed)
            if not isinstance(goal, Mapping):
                raise TypeError("read_goal() must return a mapping")
            on_frame(InputFrame(tick, now, elapsed, sensors, deepcopy(dict(goal))))
            count += 1
            tick += 1
            # Keep the original time grid even when work takes longer than a tick.
            finished = self._clock()
            if origin + tick * self.period < finished:
                tick = math.floor((finished - origin) / self.period) + 1
        return count
