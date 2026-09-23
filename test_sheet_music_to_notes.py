"""Timing integration tests; the image-routing test mocks the external OMR engine."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from music21 import bar, chord, layout, meter, note, stream, tempo, tie

from sheet_music_to_notes import musicxml_to_notes, sheet_music_to_notes


class SheetMusicTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_score(self, measures):
        part = stream.Part()
        for measure in measures:
            part.append(measure)
        score = stream.Score()
        score.insert(0, part)
        path = self.root / "score.musicxml"
        score.write("musicxml", fp=path)
        return path

    def simple_score(self, repeat=False):
        measure = stream.Measure(number=1)
        measure.append(meter.TimeSignature("4/4"))
        measure.append(note.Note("C4", quarterLength=1))
        measure.append(note.Rest(quarterLength=1))
        measure.append(note.Note("D4", quarterLength=1))
        measure.append(note.Rest(quarterLength=1))
        if repeat:
            measure.rightBarline = bar.Repeat(direction="end", times=2)
        return self.write_score([measure])

    def test_chords_ties_accidentals_and_tempo_changes(self):
        first = stream.Measure(number=1)
        first.append(meter.TimeSignature("4/4"))
        first.append(tempo.MetronomeMark(number=60))
        first.append(chord.Chord(["C4", "E4"], quarterLength=1))
        first.append(note.Rest(quarterLength=1))
        held = note.Note("G4", quarterLength=2)
        held.tie = tie.Tie("start")
        first.append(held)
        second = stream.Measure(number=2)
        second.append(tempo.MetronomeMark(number=120))
        continuation = note.Note("G4", quarterLength=1)
        continuation.tie = tie.Tie("stop")
        second.append(continuation)
        second.append(note.Note("A-4", quarterLength=1))
        second.append(note.Rest(quarterLength=2))
        actual = musicxml_to_notes(self.write_score([first, second]), bpm=200)
        self.assertEqual(actual, [(0.0, 1.0, 60), (0.0, 1.0, 64),
                                  (2.0, 2.5, 67), (4.5, 0.5, 68)])

    def test_fallback_tempo_and_rests(self):
        path = self.simple_score()
        self.assertEqual(musicxml_to_notes(path), [(0.0, 0.5, 60), (1.0, 0.5, 62)])
        self.assertEqual(musicxml_to_notes(path, bpm=60), [(0.0, 1.0, 60), (2.0, 1.0, 62)])

    def test_repeats(self):
        self.assertEqual(musicxml_to_notes(self.simple_score(repeat=True)),
                         [(0.0, 0.5, 60), (1.0, 0.5, 62),
                          (2.0, 0.5, 60), (3.0, 0.5, 62)])

    def test_image_routes_recognized_xml_to_notes(self):
        fixture = self.simple_score()
        image = self.root / "sheet with spaces.PNG"
        image.touch()

        def fake_recognition(command, **kwargs):
            output = Path(command[command.index("-output") + 1]) / "book"
            output.mkdir()
            shutil.copyfile(fixture, output / "book.musicxml")
            self.assertEqual(command[-2:], ["--", str(image.resolve())])
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("sheet_music_to_notes.shutil.which", return_value="Audiveris"), \
                patch("sheet_music_to_notes.subprocess.run", side_effect=fake_recognition):
            self.assertEqual(sheet_music_to_notes(image),
                             [(0.0, 0.5, 60), (1.0, 0.5, 62)])

    def test_invalid_tempo(self):
        for bpm in (0, -1, float("nan"), float("inf")):
            with self.subTest(bpm=bpm), self.assertRaises(ValueError):
                musicxml_to_notes(self.simple_score(), bpm=bpm)

    def test_piano_upper_staff_only_including_low_notes(self):
        right = stream.PartStaff(id="right")
        left = stream.PartStaff(id="left")
        for part, pitches in ((right, ["C3", "E3"]), (left, ["C5", "E5"])):
            measure = stream.Measure(number=1)
            measure.append(meter.TimeSignature("4/4"))
            # Even when the tempo is on the discarded staff, it must survive.
            if part is left:
                measure.append(tempo.MetronomeMark(number=60))
            measure.append(chord.Chord(pitches, quarterLength=4))
            part.append(measure)
        score = stream.Score()
        score.insert(0, right)
        score.insert(0, left)
        score.insert(0, layout.StaffGroup([right, left], symbol="brace", barTogether=True))
        path = self.root / "piano.musicxml"
        score.write("musicxml", fp=path)
        self.assertEqual(musicxml_to_notes(path), [(0.0, 4.0, 48), (0.0, 4.0, 52)])


if __name__ == "__main__":
    unittest.main()
