"""The piano keyboard: its keys and their MIDI pitches."""

NUM_KEYS = 88
"""Keys on a full piano."""

LOWEST_PITCH = 21
"""MIDI pitch of key 0, A0. Key i is pitch LOWEST_PITCH + i, up to C8 (108)."""

PIANO_PITCHES: list[int] = list(range(LOWEST_PITCH, LOWEST_PITCH + NUM_KEYS))
"""MIDI pitch of each key, in key order."""

MIDI_PITCHES = range(128)
"""Every valid MIDI pitch."""


def key_of(pitch: int) -> int:
    """The key index of a piano MIDI pitch."""
    if pitch not in PIANO_PITCHES:
        raise ValueError(f"MIDI pitch {pitch} is not on the piano (21-108)")
    return pitch - LOWEST_PITCH
