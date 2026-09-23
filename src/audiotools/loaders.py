class MidiFile:
    """
    Dummy MIDI file class
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.load()
        
    def load(self):
        with open(self.filepath, 'rb') as f:
            data = f.read()
            return data