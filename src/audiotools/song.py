class Sequence:
    def __init__(self):
        self.notes = []
    def add_note(self, pitch, start_time, duration):
        self.notes.append((pitch, start_time, duration))
    def get_notes(self):
        return self.notes
    def play(self):
        pass

class Midi:
    def __init__(self):
        pass
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.data = f.read()
    def play(self):
        pass

import pygame
# pygame for audio?
class Audio:
    def __init__(self):
        pass 
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.data = f.read()
    def play(self):
        pygame.mixer.init()
        pygame.mixer.music.load(self.data)
        pygame.mixer.music.play()

