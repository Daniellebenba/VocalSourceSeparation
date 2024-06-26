import torch
import pandas as pd
import torchaudio
import random
import numpy as np
import os

from .podcastmix_utils import PodcastMixDB, Resampler, PodcastDataSource
from nnabla.utils.data_source import DataSource



class PodcastMixDataSourceSynth(DataSource):
    dataset_name = "PodcastMix-Synth"
    data_source = PodcastDataSource.Synth
    # def __init__(self, csv_dir, sample_rate=44100, original_sample_rate=44100, segment=2,
    #              shuffle_tracks=False, multi_speakers=False):

    def __init__(self,
                 args,  # from origin MusDB
                 subset: str = "train",
                 sample_rate=44100, original_sample_rate=44100, segment=6,  # 2,
                 download=False, samples_per_track=64, source_augmentations=lambda audio: audio,
                 shuffle_tracks=False, multi_speakers=False,
                 random_track_mix=False, dtype=np.float32, seed=42, rng=None,  # from origin MusDB
                 to_stereo: bool = True):

        super(PodcastMixDataSourceSynth, self).__init__(shuffle=True)

        """
        train_set = PodcastMixDataloader(
        csv_dir=conf["data"]["train_dir"],
        sample_rate=conf["data"]["sample_rate"],
        original_sample_rate=conf["data"]["original_sample_rate"],
        segment=conf["data"]["segment"],
        shuffle_tracks=True,
        multi_speakers=conf["training"]["multi_speakers"])
        """
        # self.data_source = PodcastDataSource.synthetic
        self.csv_dir = os.path.join(args.root, "metadata", subset)
        self.root = os.path.dirname(args.root)        # added
        self.segment = segment
        # sample_rate of the original files
        self.original_sample_rate = original_sample_rate
        # destination sample_rate for resample
        self.sample_rate = sample_rate
        self.shuffle_tracks = shuffle_tracks
        self.multi_speakers = multi_speakers

        if not self.sample_rate == self.original_sample_rate:       # resample to new sample rate as needed
            self.resampler = Resampler(
                input_sr=self.original_sample_rate,
                output_sr=self.sample_rate,
                dtype=torch.float32,
                filter='hann'
            )

        # declare dataframes
        # self.speech_csv_path = os.path.join(self.csv_dir, 'speech.csv')
        # self.music_csv_path = os.path.join(self.csv_dir, 'music.csv')
        self.mix_csv_path = os.path.join(self.csv_dir, f'{subset}.csv')

        self.df_mix = pd.read_csv(self.mix_csv_path, engine='python')

        # self.df_speech = pd.read_csv(self.speech_csv_path, engine='python')

        # dictionary of speakers
        self.speakers_dict = {}
        for speaker_id in self.df_mix.speaker_id.unique():
            self.speakers_dict[speaker_id] = self.df_mix.loc[
                self.df_mix['speaker_id'] == speaker_id
                ]
        # self.df_music = pd.read_csv(self.music_csv_path, engine='python')

        # initialize indexes
        # self.speech_inxs = list(range(len(self.df_speech)))
        # self.music_inxs = list(range(len(self.df_music)))

        self.speech_inxs = list(range(len(self.df_mix)))
        self.music_inxs = list(range(len(self.df_mix)))

        # declare the resolution of the reduction factor.
        # this will create N different gain values max
        # 1/denominator_gain to multiply the music gain
        self.denominator_gain = 20
        self.gain_ramp = np.array(range(1, self.denominator_gain, 1)) / self.denominator_gain
        self.to_stereo = to_stereo

        # ---------- from origin MusDB ------------
        if rng is None:
            rng = np.random.RandomState(seed)
        self.rng = rng

        random.seed(seed)
        self.args = args
        self.download = args.root is None
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix

        self.mus = PodcastMixDB(
            data_source=self.data_source,
            root=args.root,
            is_wav=args.is_wav,
            split=None,
            subsets=subset#,        # change since train was hardcoded
            # download=download
        )
        self.sample_rate = 44100  # musdb has fixed sample rate     # TODO maybe change: the same with PodcastMix
        self.dtype = dtype

        self._size = len(self) * self.samples_per_track
        self._variables = ('mixture', 'target')
        self.reset()

        # ---------- END from origin MusDB ------------

        # shuffle the static random gain to use it in testing
        np.random.shuffle(self.gain_ramp)

        # use soundfile as backend
        torchaudio.set_audio_backend(backend='soundfile')

    def __len__(self):
        # return min([len(self.df_speech), len(self.df_music)])
        return len(self.df_mix)


    def compute_rand_offset_duration(self, original_num_frames, segment_frames):
        """ Computes a random offset and the number of frames to read from a file
        in order to get a rantom subsection of length equal to segment. If segment_frames
         is bigger than the original_num_frames, it returns
        0 for the offset and the number of frames contained in the audio. so its shorter
        than the desired segment.

        Parameters:
        - original_num_frames (int) : number of frames of the full audio file
        - segment_frames (int): number of frames of the desired segment

        Returns (tuple):
        - offset (int) : the computed random offset in frames where the audio should be
        loaded
        - segment_frames (int) : the number of frames contained in the segment.
        """
        offset = 0
        if segment_frames > original_num_frames:
            # segment: |....|
            # audio:   |abc|
            # the file is shorter than the desired segment: |abc|
            segment_frames = original_num_frames
        else:
            # segment: |....|
            # audio:   |abcdefg|
            # the file is longer than the desired segment: |cdef|
            offset = int(random.uniform(0, original_num_frames - segment_frames))

        return offset, segment_frames

    def load_mono_random_segment(self, audio_signal, audio_length, audio_path, max_segment):
        while audio_length - np.count_nonzero(audio_signal) == audio_length:
            # If there is a seg, start point is set randomly
            offset, duration = self.compute_rand_offset_duration(
                audio_length,
                max_segment
            )

            # audio_path = "/Users/daniellebenbashat/Documents/IDC/signal_processing/FinalProject/data/podcastmix/podcastmix-synth/test/music/1000069.flac"
            # audio_path = "/Users/daniellebenbashat/Documents/IDC/signal_processing/FinalProject/data/podcastmix/podcastmix-synth/test/problematic/689873.flac" # TODO Warning For tets NOW!! rmeove:
            # load the audio with the computed offsets
            audio_signal, _ = torchaudio.load(
                audio_path,
                frame_offset=offset,
                num_frames=duration
            )
            audio_signal = audio_signal.numpy()
        # convert to mono
        if len(audio_signal) == 2:
            audio_signal = np.mean(audio_signal, axis=0)[np.newaxis, ...]
        return audio_signal

    def load_non_silent_random_music(self, row):
        """ Randomly selects a non_silent part of the audio given by audio_path

        Parameters:
        - audio_path (str) : path to the audio file

        Returns:
        - audio_signal (torchaudio) : waveform of the
        """
        # info = torchaudio.info(audio_path)
        # music sample_rate
        length = int(row['length.1'])
        audio_signal = np.zeros(self.segment * self.original_sample_rate)
        # iterate until the segment is not silence
        audio_path = os.path.join(os.path.dirname(self.root), row['music_path'])
        audio_signal = self.load_mono_random_segment(audio_signal, length, audio_path,
                                                     self.segment * self.original_sample_rate)

        # zero pad if the size is smaller than seq_duration
        seq_duration_samples = int(self.segment * self.original_sample_rate)
        total_samples = audio_signal.shape[-1]
        if seq_duration_samples > total_samples:
            # add zeros at beginning and at with random offset
            padding_offset = random.randint(0, seq_duration_samples - total_samples)
            padding = np.zeros(seq_duration_samples)
            start_index = padding_offset
            padding[start_index:start_index + total_samples] = audio_signal
            audio_signal = padding

            # padding_offset = random.randint(0, seq_duration_samples - total_samples)
            # audio_signal = torch.nn.ConstantPad1d(
            #     (
            #         padding_offset,
            #         seq_duration_samples - total_samples - padding_offset
            #     ),
            #     0
            # )(audio_signal)

        return audio_signal

    # def rms(self, audio):
    #     """ computes the RMS of an audio signal
    #     """
    #     return torch.sqrt(torch.mean(audio ** 2))

    def rms(self, audio):
        """Computes the RMS of an audio signal."""
        return np.sqrt(np.mean(audio ** 2))

    def load_speechs(self, speech_idx, sample=True):
        """
        concatenates random speech files from the same speaker as speech_idx until
        obtaining a buffer with a length of at least the lenght of the
        input segment.
        If multispeaker is used, a single audio file from a different
        speaker is overlapped in a random position of the buffer, to emulate
        speakers interruptions once every 10 items.
        The buffer is shifted in a random position to prevent always getting
        buffers that starts with the beginning of a speech.
        Returns the shifted buffer with a length equal to segment.
        """
        length = 264600
        speaker_csv_id = self.df_mix.iloc[speech_idx].speaker_id
        array_size = self.original_sample_rate * self.segment
        speech_mix = np.zeros(0)
        speech_counter = 0
        while speech_counter < array_size:
            # file is shorter than segment, concatenate with more until
            # is at least the same length
            if sample:
                row_speech = self.speakers_dict[speaker_csv_id].sample().iloc[0]
            else:
                row_speech = self.df_mix.iloc[speech_idx]

            path_dir = os.path.dirname(self.root)  # todo warning change here
            audio_path = os.path.join(path_dir, row_speech['speech_path'])
            speech_signal, _ = torchaudio.load(
                audio_path
            )
            speech_signal = speech_signal.numpy()
            # add the speech to the buffer
            # speech_mix = torch.cat((speech_mix, speech_signal[0]))
            speech_mix = np.concatenate([speech_mix, speech_signal[0]])
            speech_counter += speech_signal.shape[-1]

        # we have a segment of at least self.segment length speech audio
        # from the same speaker
        if self.multi_speakers and speech_idx % 10 == 0:
            # every 10 iterations overlap another speaker
            list_of_speakers = list(self.speakers_dict.keys())
            list_of_speakers.remove(speaker_csv_id)
            non_speaker_id = random.sample(list_of_speakers, 1)[0]
            row_speech = self.speakers_dict[non_speaker_id].sample()
            audio_path = row_speech['speech_path'].values[0]
            other_speech_signal, _ = torchaudio.load(
                audio_path
            )

            other_speech_signal_length = other_speech_signal.shape[-1]
            if len(speech_mix) < other_speech_signal.shape[-1]:
                # the second speaker is longer than the original one
                other_speech_signal_length = len(speech_mix)
            offset = random.randint(0, len(speech_mix) - other_speech_signal_length)
            speech_mix[offset:offset + other_speech_signal_length] += other_speech_signal[0][
                                                                      :other_speech_signal_length]
            speech_mix = speech_mix / 2

        # we have a segment with the two speakers, the second in a random start.
        # now we randomly shift the array to pick the start
        offset = random.randint(0, array_size)
        zeros_aux = np.zeros(len(speech_mix))
        aux = speech_mix[:offset]

        zeros_aux[:len(speech_mix) - offset] = speech_mix[offset:len(speech_mix)]
        zeros_aux[len(speech_mix) - offset:] = aux

        return zeros_aux[:array_size]

    def __getitem__(self, idx):

        # if (idx == 0 and self.shuffle_tracks):
        #     # shuffle on first epochs of training and validation. Not testing
        #     random.shuffle(self.music_inxs)
        #     random.shuffle(self.speech_inxs)

        # get corresponding index from the list
        idx = idx // self.original_sample_rate

        music_idx = self.music_inxs[idx]
        speech_idx = self.speech_inxs[idx]

        # Get the row in speech dataframe
        row_music = self.df_mix.iloc[music_idx]
        sources_list = []

        # We want to cleanly separate Speech, so its the first source
        # in the sources_list
        speech_signal = self.load_speechs(speech_idx)
        music_signal = self.load_non_silent_random_music(row_music)

        # speech_signal = torch.from_numpy(speech_signal)
        # music_signal = torch.from_numpy(music_signal)

        if not self.sample_rate == self.original_sample_rate:
            speech_signal = self.resampler.forward(speech_signal)
            music_signal = self.resampler.forward(music_signal)

        # gain based on RMS in order to have RMS(speech_signal) >= RMS(music_singal)
        reduction_factor = self.rms(speech_signal) / self.rms(music_signal)
        # now we know that rms(r * music_signal) == rms(speech_signal)
        if self.shuffle_tracks:
            # random gain for training and validation
            music_gain = random.uniform(1 / self.denominator_gain, 1) * reduction_factor
        else:
            # fixed gain for testing
            music_gain = self.gain_ramp[idx % len(self.gain_ramp)] * reduction_factor

        # multiply the music by the gain factor and add to the sources_list
        music_signal = music_gain * music_signal

        # append sources:
        sources_list.append(speech_signal)
        sources_list.append(music_signal)

        # compute the mixture as the avg of both sources
        mixture = 0.5 * (sources_list[0] + sources_list[1])
        mixture = np.squeeze(mixture)

        # Stack sources
        sources = np.vstack(sources_list).astype(np.float64)       # todo: change type to float64? also need todo as in the real data

        # Convert sources to tensor
        # sources = torch.from_numpy(sources)       # we need numpy array
        mixture = mixture.astype(np.float64)
        if self.to_stereo:
            mixture = np.stack((mixture, mixture), axis=0)

        return mixture, sources

    def get_infos(self):
        """Get dataset infos (for publishing models).
        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        return infos


    def _get_data(self, position):
        return self[position]


    def get_data_by_track(self, track):
        df_mix = self.df_mix
        idx = df_mix[df_mix["speech_path"].apply(lambda x: x.split("/")[-1]) == track.path.split("/")[-1]].index[0]

        music_idx = self.music_inxs[idx]
        speech_idx = self.speech_inxs[idx]

        # Get the row in speech dataframe
        row_music = self.df_mix.iloc[music_idx]
        sources_list = []

        # We want to cleanly separate Speech, so its the first source
        # in the sources_list
        speech_signal = self.load_speechs(speech_idx, sample=False)
        music_signal = self.load_non_silent_random_music(row_music)

        # speech_signal = torch.from_numpy(speech_signal)
        # music_signal = torch.from_numpy(music_signal)

        if not self.sample_rate == self.original_sample_rate:
            speech_signal = self.resampler.forward(speech_signal)
            music_signal = self.resampler.forward(music_signal)

        # gain based on RMS in order to have RMS(speech_signal) >= RMS(music_singal)
        reduction_factor = self.rms(speech_signal) / self.rms(music_signal)
        # now we know that rms(r * music_signal) == rms(speech_signal)
        if self.shuffle_tracks:
            # random gain for training and validation
            music_gain = random.uniform(1 / self.denominator_gain, 1) * reduction_factor
        else:
            # fixed gain for testing
            music_gain = self.gain_ramp[idx % len(self.gain_ramp)] * reduction_factor

        # multiply the music by the gain factor and add to the sources_list
        music_signal = music_gain * music_signal

        # append sources:
        sources_list.append(speech_signal)
        sources_list.append(music_signal)

        # compute the mixture as the avg of both sources
        mixture = 0.5 * (sources_list[0] + sources_list[1])
        mixture = np.squeeze(mixture)

        # Stack sources
        sources = np.vstack(sources_list).astype(np.float64)       # todo: change type to float64? also need todo as in the real data

        # Convert sources to tensor
        # sources = torch.from_numpy(sources)       # we need numpy array
        mixture = mixture.astype(np.float64)
        if self.to_stereo:
            mixture = np.stack((mixture, mixture), axis=0)

        return mixture, sources

    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(PodcastMixDataSourceSynth, self).reset()


class PodcastMixDataSourceReal(DataSource):
    # path = '/Users/daniellebenbashat/Documents/IDC/signal_processing/FinalProject/data/podcastmix_data/podcastmix_data-real-with-reference'
    dataset_name = "PodcastMix-RealWithRef"
    data_source = PodcastDataSource.RealWithRef

    def __init__(self,
                 args, # from origin MusDB
                 subset: str = "metadata",
                 sample_rate=44100, segment=6, #2,
                 download=False, samples_per_track=64, source_augmentations=lambda audio: audio, random_track_mix=False, dtype=np.float32, seed=42, rng=None,     # from origin MusDB
                 to_stereo: bool = True):

        super(PodcastMixDataSourceReal, self).__init__(shuffle=True)

        csv_dir = os.path.join(args.root, "metadata")
        self.segment = segment
        self.sample_rate = sample_rate
        self.root = os.path.dirname(os.path.dirname(os.path.dirname(csv_dir)))        # added
        self.mix_csv_path = os.path.join(csv_dir, 'metadata.csv')
        self.df_mix = pd.read_csv(self.mix_csv_path, engine='python', delimiter=';')
        self.to_stereo = to_stereo

        # ---------- from origin MusDB ------------
        if rng is None:
            rng = np.random.RandomState(seed)
        self.rng = rng

        random.seed(seed)
        self.args = args
        self.download = args.root is None
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix

        self.mus = PodcastMixDB(
            data_source=self.data_source,
            root=args.root,
            # df_tracks=[self.df_mix],
            is_wav=args.is_wav,
            split=None,
            subsets="metadata",        # change since train was hardcoded
            download=download
        )
        self.sample_rate = 44100  # musdb has fixed sample rate     # TODO maybe change: the same with PodcastMix
        self.dtype = dtype

        self._size = len(self) * self.samples_per_track
        self._variables = ('mixture', 'target')
        self.reset()

        # ---------- END from origin MusDB ------------
        torchaudio.set_audio_backend(backend='soundfile')



    def __len__(self):
        return len(self.df_mix)

    def __getitem__(self, index):
        row = self.df_mix.iloc[index]
        podcast_path = os.path.join(self.root, row['mix_path'])
        speech_path = os.path.join(self.root, row['speech_path'])
        music_path = os.path.join(self.root, row['music_path'])
        sources_list = []
        start_second = 1
        mixture, _ = torchaudio.load(
            podcast_path,
            frame_offset=start_second * self.sample_rate,
            num_frames=self.segment * self.sample_rate
        )
        speech, _ = torchaudio.load(
            speech_path,
            frame_offset=start_second * self.sample_rate,
            num_frames=self.segment * self.sample_rate
        )
        music, _ = torchaudio.load(
            music_path,
            frame_offset=start_second * self.sample_rate,
            num_frames=self.segment * self.sample_rate
        )
        sources_list.append(speech[0])
        sources_list.append(music[0])
        sources = np.vstack(sources_list).astype(np.float64)       # sources.shape: (2, 88200)

        # sources = torch.from_numpy(sources)       # changed no need torch
        # changed: duplicate track to stereo support
        mixture = mixture[0].numpy().astype(np.float64)
        if self.to_stereo:
            mixture = np.stack((mixture, mixture), axis=0)
        return mixture, sources      # return numpy arrays: (2, 88200), (2, 88200)


    def _get_data(self, position):
        return self[position]


    def reset(self):
        if self._shuffle:
            self._indexes = self.rng.permutation(self._size)
        else:
            self._indexes = np.arange(self._size)
        super(PodcastMixDataSourceReal, self).reset()





