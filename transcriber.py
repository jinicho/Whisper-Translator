import threading
import collections
import queue
import numpy as np
import webrtcvad
import torch

DEFAULT_SAMPLE_RATE = 16000
AGGRESSIVENESS = 3


class LoopbackAudio(threading.Thread):
    def __init__(self, callback, mic, samplerate=DEFAULT_SAMPLE_RATE):
        threading.Thread.__init__(self)
        self.callback = callback
        self.samplerate = samplerate
        self.mic = mic
        self.stop_event = threading.Event()

    def run(self):
        with self.mic.recorder(samplerate=self.samplerate) as recorder:
            while not self.stop_event.is_set():
                data = recorder.record(numframes=640)
                self.callback(data)

    def stop(self):
        self.stop_event.set()


class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    RATE_PROCESS = DEFAULT_SAMPLE_RATE
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS):
        def proxy_callback(in_data):
            callback(in_data)

        if callback is None:
            def callback(in_data): return self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS /
                              float(self.BLOCKS_PER_SECOND))

        self.soundcard_reader = LoopbackAudio(
            callback=proxy_callback, mic=device, samplerate=self.sample_rate)
        self.soundcard_reader.daemon = True
        self.soundcard_reader.start()

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.soundcard_reader.stop()
        self.soundcard_reader.join()

    frame_duration_ms = property(
        lambda self: 1000 * self.block_size // self.sample_rate)


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, event=None):
        super().__init__(device=device, input_rate=input_rate)
        self.event = event
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            raise Exception("Resampling required")

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None:
            frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            if self.event.is_set():
                return

            mono_frame = np.mean(frame, axis=1)
            frame = np.int16(mono_frame * 32768)
            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()


def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    import six
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)
    return result


def start_listen(whisper_model, model, get_speech_ts, mic, event, callback):
    print('start listening')
    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=AGGRESSIVENESS,
                         device=mic,
                         input_rate=DEFAULT_SAMPLE_RATE,
                         event=event)

    frames = vad_audio.vad_collector()

    wav_data = bytearray()
    for frame in frames:
        if frame is not None:
            wav_data.extend(frame)
        else:
            newsound = np.frombuffer(wav_data, np.int16)
            audio_float32 = Int2Float(newsound)
            time_stamps = get_speech_ts(
                audio_float32, model, sampling_rate=DEFAULT_SAMPLE_RATE)

            if (len(time_stamps) > 0):
                transcript = whisper_model.transcribe(audio=audio_float32)
                translation = translate_text('ko', transcript['text'])
                callback(transcript['text'], translation['translatedText'])
            else:
                pass
            print()

            wav_data = bytearray()
    print('listening stoped')
    vad_audio.destroy()


def Int2Float(sound):
    _sound = np.copy(sound)  #
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype('float32')
    if abs_max > 0:
        _sound *= 1/abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32
