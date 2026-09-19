from .song import Midi, Audio, NoteSequence

def midi_to_sequence(midi: Midi) -> NoteSequence:
    # something like this but probably slightly different
    new_sequence = NoteSequence()
    # this is just a placeholder basic idea
    for note in midi.notes:
        pitch = note.pitch
        start_time = note.start_time
        duration = note.duration
        new_sequence.add_note(pitch, start_time, duration)
    return new_sequence

def audio_to_midi(audio: Audio) -> Midi:
    new_midi = Midi()
    # import some library to do this probably
    
    return new_midi