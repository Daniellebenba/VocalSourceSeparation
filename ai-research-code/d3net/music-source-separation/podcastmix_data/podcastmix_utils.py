from scipy import special
import warnings
from typing import List
import collections
import torch
import pandas as pd
from os import path as op
import numpy as np
import yaml
import musdb
import os
from enum import Enum
import stempeg
from museval import TrackStore, evaluate

class PodcastDataSource(Enum):
    RealWithRef = "real_with_ref"
    Synth = "synth"


mus_podcast_track_mapping = {"vocals": "speech", "music": "music"}


class Resampler(torch.nn.Module):
    """
    Efficiently resample audio signals
    This module is much faster than resampling with librosa because
    it exploits pytorch's efficient conv1d operations
    This module is also faster than the existing pytorch resample function in
    https://github.com/pytorch/audio/blob/b6a61c3f7d0267c77f8626167cc1eda0335f2753/torchaudio/compliance/kaldi.py#L892
    Based on
    https://github.com/danpovey/filtering/blob/master/lilfilter/resampler.py
    with improvements to include additional filter types and input
    parameters that align with the librosa api
    """

    def __init__(self,
                 input_sr, output_sr, dtype,
                 num_zeros=64,
                 cutoff_ratio=0.95,
                 filter='kaiser',
                 beta=14.0):
        super().__init__()  # init the base class
        """
        This creates an object that can apply a symmetric FIR filter
        based on torch.nn.functional.conv1d.

        Args:
          input_sr:  The input sampling rate, AS AN INTEGER..
              does not have to be the real sampling rate but should
              have the correct ratio with output_sr.
          output_sr:  The output sampling rate, AS AN INTEGER.
              It is the ratio with the input sampling rate that is
              important here.
          dtype: The torch dtype to use for computations
               (would be preferrable to
               set things up so passing the dtype isn't necessary)
          num_zeros: The number of zeros per side in the (sinc*hanning-window)
              filter function.  More is more accurate, but 64 is already
              quite a lot. The kernel size is 2*num_zeros + 1.
          cutoff_ratio: The filter rolloff point as a fraction of the
             Nyquist frequency.
          filter: one of ['kaiser', 'kaiser_best', 'kaiser_fast', 'hann']
          beta: parameter for 'kaiser' filter

        You can think of this algorithm as dividing up the signals
        (input,output) into blocks where there are `input_sr` input
        samples and `output_sr` output samples.  Then we treat it
        using convolutional code, imagining there are `input_sr`
        input channels and `output_sr` output channels per time step.

        """
        assert isinstance(input_sr, int) and isinstance(output_sr, int)
        if input_sr == output_sr:
            self.resample_type = 'trivial'
            return

        def gcd(a, b):
            """ Return the greatest common divisor of a and b"""
            assert isinstance(a, int) and isinstance(b, int)
            if b == 0:
                return a
            else:
                return gcd(b, a % b)

        d = gcd(input_sr, output_sr)
        input_sr, output_sr = input_sr // d, output_sr // d

        assert dtype in [torch.float32, torch.float64]
        assert num_zeros > 3  # a reasonable bare minimum
        np_dtype = np.float32 if dtype == torch.float32 else np.float64

        assert filter in ['hann', 'kaiser', 'kaiser_best', 'kaiser_fast']

        if filter == 'kaiser_best':
            num_zeros = 64
            beta = 14.769656459379492
            cutoff_ratio = 0.9475937167399596
            filter = 'kaiser'
        elif filter == 'kaiser_fast':
            num_zeros = 16
            beta = 8.555504641634386
            cutoff_ratio = 0.85
            filter = 'kaiser'

        # Define one 'block' of samples `input_sr` input samples
        # and `output_sr` output samples.  We can divide up
        # the samples into these blocks and have the blocks be
        # in correspondence.

        # The sinc function will have, on average, `zeros_per_block`
        # zeros per block.
        zeros_per_block = min(input_sr, output_sr) * cutoff_ratio

        # The convolutional kernel size will be n = (blocks_per_side*2 + 1),
        # i.e. we add that many blocks on each side of the central block.  The
        # window radius (defined as distance from center to edge)
        # is `blocks_per_side` blocks.  This ensures that each sample in the
        # central block can "see" all the samples in its window.
        #
        # Assuming the following division is not exact, adding 1
        # will have the same effect as rounding up.
        # blocks_per_side = 1 + int(num_zeros / zeros_per_block)
        blocks_per_side = int(np.ceil(num_zeros / zeros_per_block))

        kernel_width = 2*blocks_per_side + 1

        # We want the weights as used by torch's conv1d code; format is
        #  (out_channels, in_channels, kernel_width)
        # https://pytorch.org/docs/stable/nn.functional.html
        weights = torch.tensor((
            output_sr,
            input_sr,
            kernel_width
        ), dtype=dtype)

        # Computations involving time will be in units of 1 block.
        # Actually this is the same as the `canonical` time axis
        # since each block has input_sr
        # input samples, so it would be one of whatever time unit we are using
        window_radius_in_blocks = blocks_per_side

        # The `times` below will end up being the args to the sinc
        #  function.
        # For the shapes of the things below, look at the args to
        # `view`.  The terms
        # below will get expanded to shape (output_sr, input_sr,
        # kernel_width) through broadcasting
        # We want it so that, assuming input_sr == output_sr,
        # along the diagonal of the central block we have t == 0.
        # The signs of the output_sr and input_sr terms need to be
        # opposite.  The
        # sign that the kernel_width term needs to be will depend
        # on whether it's
        # convolution or correlation, and the logic is tricky.. I
        # will just find
        # which sign works.

        times = (
            np.arange(output_sr, dtype=np_dtype).reshape(
                (output_sr, 1, 1)
            ) / output_sr -
            np.arange(input_sr, dtype=np_dtype).reshape(
                (1, input_sr, 1)
            ) / input_sr -
            (np.arange(kernel_width, dtype=np_dtype).reshape(
                (1, 1, kernel_width)
            ) - blocks_per_side))

        def hann_window(a):
            """
            hann_window returns the Hann window on [-1,1], which is zero
            if a < -1 or a > 1, and otherwise 0.5 + 0.5 cos(a*pi).
            This is applied elementwise to a, which should be a NumPy array.

            The heaviside function returns (a > 0 ? 1 : 0).
            """
            return np.heaviside(
                1 - np.abs(a),
                0.0
            ) * (0.5 + 0.5 * np.cos(a * np.pi))

        def kaiser_window(a, beta):
            w = special.i0(beta * np.sqrt(
                np.clip(1 - ((a - 0.0) / 1.0) ** 2.0, 0.0, 1.0)
            )) / special.i0(beta)
            return np.heaviside(1 - np.abs(a), 0.0) * w

        # The weights below are a sinc function times a Hann-window function.
        # Multiplication by zeros_per_block normalizes the sinc function
        # (to compensate for scaling on the x-axis), so that the integral is 1.
        # Division by input_sr normalizes the input function. Think of the
        # input
        # as a stream of dirac deltas passing through a low pass filter:
        # in order to have the same magnitude as the original input function,
        # we need to divide by the number of those deltas per unit time.
        if filter == 'hann':
            weights = (np.sinc(times * zeros_per_block)
                       * hann_window(times / window_radius_in_blocks)
                       * zeros_per_block / input_sr)
        else:
            weights = (np.sinc(times * zeros_per_block)
                       * kaiser_window(times / window_radius_in_blocks, beta)
                       * zeros_per_block / input_sr)

        self.input_sr = input_sr
        self.output_sr = output_sr

        # weights has dim (output_sr, input_sr, kernel_width).
        # If output_sr == 1, we can fold the input_sr into the
        # kernel_width (i.e. have just 1 input channel); this will make the
        # convolution faster and avoid unnecessary reshaping.

        assert weights.shape == (output_sr, input_sr, kernel_width)
        if output_sr == 1:
            self.resample_type = 'integer_downsample'
            self.padding = input_sr * blocks_per_side
            weights = torch.tensor(weights, dtype=dtype, requires_grad=False)
            self.weights = weights.transpose(1, 2).contiguous().view(
                1, 1, input_sr * kernel_width
            )

        elif input_sr == 1:
            # In this case we'll be doing conv_transpose, so we want
            # the same weights that
            # we would have if we were *downsampling* by this factor
            # -- i.e. as if input_sr,
            # output_sr had been swapped.
            self.resample_type = 'integer_upsample'
            self.padding = output_sr * blocks_per_side
            weights = torch.tensor(weights, dtype=dtype, requires_grad=False)
            self.weights = weights.flip(2).transpose(0, 2).contiguous().view(
                1, 1, output_sr * kernel_width
            )
        else:
            self.resample_type = 'general'
            self.reshaped = False
            self.padding = blocks_per_side
            self.weights = torch.tensor(
                weights,
                dtype=dtype,
                requires_grad=False)

        self.weights = torch.nn.Parameter(
            self.weights,
            requires_grad=False)

    @torch.no_grad()
    def forward(self, data):
        """
        Resample the data

        Args:
         input: a torch.Tensor with the same dtype as was passed to the
           constructor.
         There must be 2 axes, interpreted as
         (minibatch_size, sequence_length)...
         the minibatch_size may in practice be the number of channels.

        Return:  Returns a torch.Tensor with the same dtype as the input, and
         dimension (minibatch_size, (sequence_length//input_sr)*output_sr),
         where input_sr and output_sr are the corresponding constructor args,
         modified to remove any common factors.
        """
        if self.resample_type == 'trivial':
            return data
        elif self.resample_type == 'integer_downsample':
            (minibatch_size, seq_len) = data.shape
            # will be shape (minibatch_size, in_channels, seq_len)
            # with in_channels == 1
            data = data.unsqueeze(1)
            data = torch.nn.functional.conv1d(
                data,
                self.weights,
                stride=self.input_sr,
                padding=self.padding)
            # shape will be (minibatch_size, out_channels = 1, seq_len);
            # return as (minibatch_size, seq_len)
            return data.squeeze(1)

        elif self.resample_type == 'integer_upsample':
            data = data.unsqueeze(1)
            data = torch.nn.functional.conv_transpose1d(
                data,
                self.weights,
                stride=self.output_sr,
                padding=self.padding)

            return data.squeeze(1)
        else:
            assert self.resample_type == 'general'
            (minibatch_size, seq_len) = data.shape
            num_blocks = seq_len // self.input_sr
            if num_blocks == 0:
                # TODO: pad with zeros.
                raise RuntimeError("Signal is too short to resample")
            # data = data[:, 0:(num_blocks*self.input_sr)]  # Truncate input
            data = data[:, 0:(num_blocks*self.input_sr)].view(
                minibatch_size,
                num_blocks,
                self.input_sr)

            # Torch's conv1d expects input data with shape (minibatch,
            # in_channels, time_steps), so transpose
            data = data.transpose(1, 2)
            data = torch.nn.functional.conv1d(
                data,
                self.weights,
                padding=self.padding)

            assert data.shape == (minibatch_size, self.output_sr, num_blocks)
            return data.transpose(1, 2).contiguous().view(
                minibatch_size, num_blocks * self.output_sr
            )


class PodcastMixDB(object):
    """
    The musdb DB Object implemented for PodcastMix data !!

    Parameters
    ----------
    root : str, optional
        musdb Root path. If set to `None` it will be read
        from the `MUSDB_PATH` environment variable

    subsets : str or list, optional
        select a _musdb_ subset `train` or `test` (defaults to both)

    is_wav : boolean, optional
        expect subfolder with wav files for each source instead stems,
        defaults to `False`

    download : boolean, optional
        download sample version of MUSDB18 which includes 7s excerpts,
        defaults to `False`

    subsets : list[str], optional
        select a _musdb_ subset `train` or `test`.
        Default `None` loads `['train', 'test']`

    split : str, optional
        when `subsets=train`, `split` selects the train/validation split.
        `split='train' loads the training split, `split='valid'` loads the validation
        split. `split=None` applies no splitting.

    Attributes
    ----------
    setup_file : str
        path to yaml file. default: `setup.yaml`
    root : str
        musdb Root path. Default is `MUSDB_PATH`. In combination with
        `download`, this path will set the download destination and set to
        '~/musdb/' by default.
    sources_dir : str
        path to Sources directory
    sources_names : list[str]
        list of names of available sources
    targets_names : list[str]
        list of names of available targets
    setup : Dict
        loaded yaml configuration
    sample_rate : Optional(Float)
        sets sample rate for optional resampling. Defaults to none
        which results in `44100.0`

    Methods
    -------
    load_mus_tracks()
        Iterates through the musdb folder structure and
        returns ``Track`` objects

    """

    def __init__(
        self,
        root,
        data_source: PodcastDataSource,
        # df_tracks: List[pd.DataFrame],      # todo change to rwad inside the df
        setup_file=None,
        is_wav=False,
        download=False,
        subsets=["train", "test"],      # can be also ["metadata"] for the test with ref dataset
        split=None,
        sample_rate=None,
    ):
        # if root is None:
        #     if download:
        #         self.root = os.path.expanduser("~/MUSDB18/MUSDB18-7")
        #     else:
        #         if "MUSDB_PATH" in os.environ:
        #             self.root = os.environ["MUSDB_PATH"]
        #         else:
        #             raise RuntimeError("Variable `MUSDB_PATH` has not been set.")
        # else:
        #     self.root = os.path.expanduser(root)
        self.data_source = data_source
        self.root = os.path.expanduser(root)
        # self.root = os.path.dirname(os.path.dirname(self.root))

        if setup_file is not None:
            setup_path = op.join(self.root, setup_file)
        else:
            setup_path = os.path.join(musdb.__path__[0], "configs", "mus.yaml")

        with open(setup_path, "r") as f:
            self.setup = yaml.safe_load(f)

        if download:        # todo suport for podcast download
            # self.url = self.setup["sample-url"]
            # self.download()
            # if not self._check_exists():
            #     raise RuntimeError(
            #         "Dataset not found."
            #         + "You can use download=True to download a sample version of the dataset"
            #     )
            raise Exception("Error not supported for now dataset downloading")

        self.sample_rate = sample_rate
        # self.setup["sources"] = {'speech': 'speech', 'music': 'music'}
        self.setup["sources"] = {'vocals': 'vocals', 'music': 'music'}
        # self.setup["targets"] = {'speech': {'speech': 1}}
        self.setup["targets"] = {'vocals': {'vocals': 1}}
        self.setup["stem_ids"] = {'mix': 0, 'vocals': 1, 'music': 2}        # {'mixture': 0, 'drums': 1, 'bass': 2, 'other': 3, 'vocals': 4}

        self.sources_names = list(self.setup["sources"].keys())
        self.targets_names = list(self.setup["targets"].keys())

        self.is_wav = is_wav
        self.tracks = self.load_tracks(subsets=subsets, split=split)

    def __getitem__(self, index):
        return self.tracks[index]

    def __len__(self):
        return len(self.tracks)

    def create_targets(self, track):
        # add targets to track
        targets = collections.OrderedDict()
        for name, target_srcs in list(self.setup["targets"].items()):
            # add a list of target sources
            target_sources = []
            for source, gain in list(target_srcs.items()):
                if source in list(track.sources.keys()):
                    # add gain to source tracks
                    track.sources[source].gain = float(gain)
                    # add tracks to components
                    target_sources.append(track.sources[source])
                    # add sources to target
            if target_sources:
                targets[name] = musdb.Target(track, sources=target_sources, name=name)

        return targets

    def load_tracks(self, subsets=None, split=None):
        """Parses the musdb folder structure, returns list of `Track` objects

        Parameters
        ==========
        subsets : list[str], optional
            select a _musdb_ subset `train` or `test`.
            Default `None` loads [`train, test`].
        split : str
            for subsets='train', `split='train` applies a train/validation split.
            if `split='valid`' the validation split of the training subset will be used


        Returns
        -------
        list[Track]
            return a list of ``Track`` Objects
        """

        if subsets is not None:
            if isinstance(subsets, str):
                subsets = [subsets]
        else:
            subsets = ["train", "test"]

        if subsets != ["train"] and split is not None:
            raise RuntimeError("Subset has to set to `train` when split is used")

        if self.data_source is PodcastDataSource.Synth:
            tracks = self.get_tracks_synth(subsets)

        elif self.data_source is PodcastDataSource.RealWithRef:
            tracks = self.get_tracks_real(subsets)

        else:
            raise Exception("not supported podcastmix data source")

        return tracks

    def get_tracks_synth(self, subsets):
        tracks = []
        valid_counter = 0
        for subset in subsets:
            valid_counter = 0
            metadata_file = op.join(self.root, "metadata", subset, f"{subset}.csv")  # todo here change if not metadata then add subdir based on subset
            if not os.path.exists(metadata_file):
                raise Exception(f"metadata file not found at {metadata_file} ")
            df = pd.read_csv(metadata_file, engine='python')  # create .csv files of the mixes for synthetic

            path_dir = os.path.dirname(os.path.dirname(self.root))  # todo warning change here
            path_dir = path_dir.replace("/podcastmix/", "/")
            for i, row in df.iterrows():
                track = musdb.MultiTrack(
                    name=row["speech_ID"]+"_"+row["name"],
                    path=os.path.join(path_dir, row["speech_path"]),
                    subset=subset,
                    is_wav=self.is_wav,
                    stem_id=self.setup["stem_ids"]["mix"],
                    sample_rate=self.sample_rate,
                )

                # add sources to track
                sources = {}
                for src in self.sources_names:
                    key = mus_podcast_track_mapping[src]
                    path = os.path.join(path_dir, row[f"{key}_path"])  # todo: add absolute path
                    # create source object
                    if os.path.exists(path):
                        sources[src] = musdb.Source(
                            track,
                            name=src,
                            path=path,
                            stem_id=self.setup["stem_ids"]["mix"],
                            sample_rate=self.sample_rate,
                        )
                    else:
                        print(f"Warning file {path} not found!!")
                track.sources = sources
                track.targets = self.create_targets(track)
                if len(sources) == len(self.sources_names):
                    valid_counter += 1
                    # add track to list of tracks
                tracks.append(track)
            print(f"Finished loading {valid_counter} valid tracks")
        return tracks


    def get_tracks_real(self, subsets):
        tracks = []
        valid_counter = 0
        assert len(subsets) == 1
        subset = subsets[0]
        subset = "metadata"
        path = op.join(self.root, subset, f"{subset}.csv")      # todo here change if not metadata then add subdir based on subset
        if not os.path.exists(path):
            raise Exception(f"metadata file not found at {path}")
        df = pd.read_csv(path, engine='python', delimiter=';')         # create .csv files of the mixes for synthetic

        path_dir = os.path.dirname(os.path.dirname(self.root))     # todo warning change here
        path_dir = path_dir.replace("/podcastmix/", "/")

        for i, row in df.iterrows():
            track = musdb.MultiTrack(
                name=row["song"],
                path=os.path.join(path_dir, row["mix_path"]),
                subset=subset,
                is_wav=self.is_wav,
                stem_id=self.setup["stem_ids"]["mix"],
                sample_rate=self.sample_rate,
            )

            # add sources to track
            sources = {}
            for src in self.sources_names:
                key = mus_podcast_track_mapping[src]
                path = os.path.join(path_dir, row[f"{key}_path"])       # todo: add absolute path
                # create source object
                if os.path.exists(path):
                    sources[src] = musdb.Source(
                        track,
                        name=src,
                        path=path,
                        # stem_id=self.setup["stem_ids"][src],        # todo here change the index stem to 0
                        stem_id=self.setup["stem_ids"]["mix"],        # todo here change the index stem to 0
                        sample_rate=self.sample_rate,
                    )
                else:
                    print(f"Warning file {path} not found!!")
            track.sources = sources
            track.targets = self.create_targets(track)
            if len(sources) == len(self.sources_names):
                valid_counter += 1

                # add track to list of tracks
                tracks.append(track)
        print(f"Finished loading {valid_counter} valid tracks")
        return tracks

def mono_to_stereo(x):
    if len(x.shape) == 1:  # mono to stereo
        x = np.stack((x, x))
    return x





def save_estimates(user_estimates, track, estimates_dir, write_stems=False):
    """Writes `user_estimates` to disk while recreating the musdb file structure in that folder.

    Parameters
    ==========
    user_estimates : Dict[np.array]
        the target estimates.
    track : Track,
        musdb track object
    estimates_dir : str,
        output folder name where to save the estimates.
    """

    track_estimate_dir = os.path.join(estimates_dir, track.subset, track.name)
    if not os.path.exists(track_estimate_dir):
        os.makedirs(track_estimate_dir)

    # write out tracks to disk
    if write_stems:
        pass
        # to be implemented
    else:
        for target, estimate in list(user_estimates.items()):
            target_path = os.path.join(track_estimate_dir, target + ".wav")
            stempeg.write_audio(
                path=target_path, data=estimate, sample_rate=track.rate
            )



def eval_podcast_track(track, user_estimates, output_dir=None, mode="v4", win=1.0, hop=1.0):
    """Compute all bss_eval metrics for the musdb track and estimated signals,
    given by a `user_estimates` dict.

    Parameters
    ----------
    track : Track
        musdb track object loaded using musdb
    estimated_sources : Dict
        dictionary, containing the user estimates as np.arrays.
    output_dir : str
        path to output directory used to save evaluation results. Defaults to
        `None`, meaning no evaluation files will be saved.
    mode : str
        bsseval version number. Defaults to 'v4'.
    win : int
        window size in

    Returns
    -------
    scores : TrackStore
        scores object that holds the framewise and global evaluation scores.
    """

    audio_estimates = []
    audio_reference = []

    # make sure to always build the list in the same order
    # therefore track.targets is an OrderedDict
    eval_targets = []  # save the list of target names to be evaluated
    for key, target in list(track.targets.items()):
        try:
            # try to fetch the audio from the user_results of a given key
            user_estimates[key]
        except KeyError:
            # ignore wrong key and continue
            continue

        # append this target name to the list of target to evaluate
        eval_targets.append(key)

    data = TrackStore(win=win, hop=hop, track_name=track.name)

    # check if vocals and accompaniment is among the targets
    has_acc = all(x in eval_targets for x in ["vocals", "accompaniment"])
    if has_acc:
        # remove accompaniment from list of targets, because
        # the voc/acc scenario will be evaluated separately
        eval_targets.remove("accompaniment")

    if len(eval_targets) >= 1:#2:
        # compute evaluation of remaining targets
        for target in eval_targets:
            audio_estimates.append(user_estimates[target][:, 0][np.newaxis, ...].T)
            audio_reference.append(track.targets[target].audio[np.newaxis, ...].T)

        SDR, ISR, SIR, SAR = evaluate(
            audio_reference,
            audio_estimates,
            win=int(win * track.rate),
            hop=int(hop * track.rate),
            mode=mode,
        )

        # iterate over all evaluation results except for vocals
        for i, target in enumerate(eval_targets):
            if target == "vocals" and has_acc:
                continue

            values = {
                "SDR": SDR[i].tolist(),
                "SIR": SIR[i].tolist(),
                "ISR": ISR[i].tolist(),
                "SAR": SAR[i].tolist(),
            }

            data.add_target(target_name=target, values=values)
    elif not has_acc:
        warnings.warn(
            UserWarning(
                "Incorrect usage of BSSeval : at least two estimates must be provided. Target score will be empty."
            )
        )

    # add vocal accompaniment targets later
    if has_acc:
        # add vocals and accompaniments as a separate scenario
        eval_targets = ["vocals", "accompaniment"]

        audio_estimates = []
        audio_reference = []

        for target in eval_targets:
            audio_estimates.append(user_estimates[target])
            audio_reference.append(track.targets[target].audio)

        SDR, ISR, SIR, SAR = evaluate(
            audio_reference,
            audio_estimates,
            win=int(win * track.rate),
            hop=int(hop * track.rate),
            mode=mode,
        )

        # iterate over all targets
        for i, target in enumerate(eval_targets):
            values = {
                "SDR": SDR[i].tolist(),
                "SIR": SIR[i].tolist(),
                "ISR": ISR[i].tolist(),
                "SAR": SAR[i].tolist(),
            }

            data.add_target(target_name=target, values=values)

    if output_dir:
        # validate against the schema
        data.validate()

        try:
            subset_path = op.join(output_dir, track.subset)

            if not op.exists(subset_path):
                os.makedirs(subset_path)

            with open(op.join(subset_path, track.name) + ".json", "w+") as f:
                f.write(data.json)

        except IOError:
            pass

    return data
