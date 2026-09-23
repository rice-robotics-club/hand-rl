This code owns the conversion of audio-->midi-->a nicer format!

This code owns:
1. an abstract base class for a song object that can either be a sequence of notes, an audio file, or a midi file with a clear label on what type it is 
2. file loaders for midi and audio files that return their raw data
3. a chunk of code that handles the actual conversion process between the three types of files