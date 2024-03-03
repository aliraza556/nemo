from contextlib import contextmanager
import os
import tempfile
import urllib.request
from pydub import AudioSegment


class DownloadService:
    @contextmanager
    def download(self, url):
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, 'temp_file')
            urllib.request.urlretrieve(url, filepath)
            audio = AudioSegment.from_file(filepath)
            wav_filepath = os.path.join(os.path.dirname(filepath), 'temp_file' + '.wav')
            audio.export(wav_filepath, format='wav')
            yield wav_filepath