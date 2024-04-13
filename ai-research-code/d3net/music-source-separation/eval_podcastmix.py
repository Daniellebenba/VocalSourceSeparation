# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
MSS evaluation code on MUSDB18 test dataset.
'''

import os
import argparse
import multiprocessing
import functools
import tqdm
import numpy as np
import yaml
import museval
import musdb
import nnabla as nn
from nnabla.ext_utils import get_extension_context
import librosa
import nnabla.functions as F

from filter import apply_mwf
from util import stft2time_domain, model_separate
from podcastmix_data.podcastmix_utils import PodcastMixDB, PodcastDataSource, save_estimates, eval_podcast_track
from podcastmix_data.podcastmix_nnbala_datasources import PodcastMixDataSourceReal, PodcastMixDataSourceSynth
from podcastmix_data.podcastmix_utils import mono_to_stereo
from model import spectogram



def separate_and_evaluate(
    track,
    model_dir,
    model_path,
    targets,
    output_dir,
    data_source: PodcastDataSource,
    dataset=None
):
    fft_size, hop_size, n_channels = 4096, 1024, 2
    if data_source is PodcastDataSource.Synth:
        assert dataset is not None
        # data_source = PodcastMixDataSourceSynth()
        # audio = track.audio[..., np.newaxis]
        audio, sources = dataset.get_data_by_track(track)
        audio = audio[..., np.newaxis]


    elif data_source is PodcastDataSource.RealWithRef:
        audio = track.audio[..., np.newaxis]
        sources = np.vstack([track.targets[t].audio for t in targets])

    else:
        raise Exception(f"Data source not supported, PodcastMix Real or Synthetic only")

    for i in range(audio.shape[1]):
        stft = librosa.stft(audio[:, i].flatten(),
                            n_fft=fft_size, hop_length=hop_size).transpose()
        if i == 0:
            data = np.ndarray(shape=(stft.shape[0], n_channels, fft_size // 2 + 1),
                              dtype=np.complex64)
        data[:, i, :] = stft

    if n_channels == 2 and audio.shape[1] == 1:
        data[:, 1] = data[:, 0]

    inp_stft = data

    out_stfts = {}
    inp_stft_contiguous = np.abs(np.ascontiguousarray(inp_stft))

    for target in targets:
        # Load the model weights for corresponding target
        # model_path = os.path.abspath(f"{os.path.join(model_dir, target)}.h5")
        # model_path = model_dir
        nn.load_parameters(model_path)
        print(f"Loaded Parameters {model_path}")
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"configs/{target}.yaml")) as file:
            # Load target specific Hyper parameters
            hparams = yaml.load(file, Loader=yaml.FullLoader)
        target_scope = f"VocalSourceSeparation/ai-research-code/d3net/music-source-separation/configs/{target}"
        target_scope = 'VocalSourceSeparation/ai-research-code/d3net/music-source-separation/configs/VocalSourceSeparation/ai-research-code/d3net/music-source-separation/configs/vocals'
        # target_scope = 'vocals'     # for the MusDB weights
        # target_scope = 'XXX'     # for the random weights
        with nn.parameter_scope(target_scope):
            out_sep = model_separate(
                inp_stft_contiguous, hparams, ch_flip_average=True)
            out_stfts[target] = out_sep * np.exp(1j * np.angle(inp_stft))

    out_stfts = apply_mwf(out_stfts, inp_stft)


    estimates = {}
    for i, target in enumerate(targets):
        estimates[target] = stft2time_domain(out_stfts[target], hop_size)   # -> (264192,2)

        # target_audio = nn.Variable([args.batch_size] + list(train_source._get_data(0)[1].shape))  # MusdDB: (batch, 2, 264600)
        # target_spec = spectogram(*stft(target_audio, n_fft=hparams['fft_size'], n_hop=hparams['hop_size'], patch_length=256),
        #                          mono=(hparams['n_channels'] == 1))          # original MusDB: mix_spec.shape: (6, 256, 2, 2049), 2: vocals (=target) and others tracks
        # loss = F.mean(F.squared_error(out_stfts[target], sources[0]))
        # Assuming out_stfts[target] and sources[0] are NumPy arrays
        # mse = np.mean(np.square(estimates[target] - sources[i]))

    if output_dir:
        save_estimates(estimates, track, output_dir)

    # scores = museval.eval_mus_track(
    scores, mse, isdr = eval_podcast_track(
        track, estimates, output_dir=output_dir
    )
    return scores, mse, isdr


if __name__ == '__main__':
    FLAG_LOCAL = False  # TODO: warning for test locally, change to False when running

    # Evaluation settings
    parser = argparse.ArgumentParser(
        description='PodcastMix Evaluation', add_help=False)

    parser.add_argument('--data-source', type=str, default="podcastmix_real",
                        help='Choose data source: podcastmix_real / podcastmix_synth')
    parser.add_argument('--model-dir', '-m', type=str,
                        default='./d3net-mss', help='path to the directory of pretrained models.')
    parser.add_argument('--model-path', '-mp', type=str,
                        default='./d3net-mss/vocals.h5', help='path to the pretrained model.')
    parser.add_argument('--targets', nargs='+', default=['vocals'],
                        type=str, help='provide targets to be processed. If none, all available targets will be computed')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Path to save musdb estimates and museval results')
    parser.add_argument('--root', type=str, help='Path to PodcastMix')

    parser.add_argument('--subset', type=str, default='test',
                        help='PodcastMix subset (`train`/`val`/`test`), valid only for Synthetic')        # if is real won't change nothing
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--is-wav', action='store_true',
                        default=False, help='flag: wav version of the dataset')
    parser.add_argument('--context', default='cudnn',
                        type=str, help='Execution on CUDA')
    args, _ = parser.parse_known_args()

    if FLAG_LOCAL:
        args.context = 'cpu'
    # Set NNabla context and Dynamic graph execution
    ctx = get_extension_context(args.context)

    if FLAG_LOCAL:
        ctx.device_id = '0'

    nn.set_default_context(ctx)

    if FLAG_LOCAL:
        args.root = "/Users/daniellebenbashat/Documents/IDC/signal_processing/FinalProject/data/podcastmix/podcastmix-real-with-reference"
        args.data_source = "podcastmix_real"

        args.root = "/Users/daniellebenbashat/Documents/IDC/signal_processing/FinalProject/data/podcastmix/podcastmix-synth"
        args.data_source = "podcastmix_synth"
        args.subset = "test"
        args.model_dir = "/Users/daniellebenbashat/PycharmProjects/audio/ai-research-code/d3net/music-source-separation/assets/"
        args.model_path = "/Users/daniellebenbashat/PycharmProjects/audio/ai-research-code/d3net/music-source-separation/assets/vocals.h5"
        args.model_path = "/Users/daniellebenbashat/PycharmProjects/audio/ai-research-code/d3net/music-source-separation/assets/vocals_run2_epoch10.h5"

    if args.data_source == "podcastmix_synth":
        # mus = PodcastMixDB(
        #     data_source=PodcastDataSource.Synth,
        #     root=args.root,
        #     download=args.root is None,
        #     subsets="test",
        #     is_wav=args.is_wav
        # )
        data_source = PodcastDataSource.Synth
        test_dataset = PodcastMixDataSourceSynth(subset=args.subset, random_track_mix=False, args=args, samples_per_track=1, to_stereo=False)
        mus = test_dataset.mus

    elif args.data_source == "podcastmix_real":
        mus = PodcastMixDB(
            data_source=PodcastDataSource.RealWithRef,
            root=args.root,
            download=args.root is None,
            subsets=args.subset,
            is_wav=args.is_wav
        )
        data_source = PodcastDataSource.RealWithRef
        test_dataset = None
    else:
        raise Exception(f"data source not supported: podcastmix_synth or podcastmix_real solely")

    # if args.data_source == "podcastmix_synth":
    #     train_dataset = PodcastMixDataSourceSynth(
    #         random_track_mix=True, args=args)
    # elif args.data_source == "podcastmix_real":
    #     train_dataset = PodcastMixDataSourceReal(
    #         random_track_mix=True, args=args)
    # else:
    #     raise Exception("not supported data source, choose musdb/ podcastmix_real/ podcastmix_synth only!")
    mse_list = []
    isdr_list = []
    if args.cores > 1:
        pool = multiprocessing.Pool(args.cores)
        results = museval.EvalStore()
        scores_list, mse_list, isdr_list = list(
            pool.imap_unordered(
                func=functools.partial(
                    separate_and_evaluate,
                    model_dir=args.model_dir,
                    model_path=args.model_path,
                    targets=args.targets,
                    output_dir=args.out_dir,
                    data_source=data_source,
                    dataset=test_dataset
                ),
                iterable=mus.tracks,
                chunksize=1
            )
        )
        pool.close()
        pool.join()
        for scores in scores_list:
            results.add_track(scores)
    else:
        results = museval.EvalStore()
        for i, track in tqdm.tqdm(enumerate(mus.tracks)):     # here iter on the datasource not tracks
            scores, mse, isdr = separate_and_evaluate(
                track=track,
                model_dir=args.model_dir,
                model_path=args.model_path,
                targets=args.targets,
                output_dir=args.out_dir,
                data_source=data_source,
                dataset=test_dataset
            )
            results.add_track(scores)
            mse_list.append(mse)
            isdr_list.append(isdr)

    print(results)
    print(f"Average MSE: {np.array(mse_list).mean()}")
    print(f"Average ISDR: {np.array(isdr_list).mean()}")
    method = museval.MethodStore()
    method.add_evalstore(results, args.model_dir)
    method.save(args.model_dir + '.pandas')


# import os
# import random
# import soundfile as sf
# import torch
# import yaml
# import json
# import argparse
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import sys
# from utils.my_import import my_import
#
# from asteroid.metrics import get_metrics
# from pytorch_lightning import seed_everything
# from PodcastMixDataloader import PodcastMixDataloader
# from asteroid.metrics import MockWERTracker
#
# seed_everything(1, workers=True)
#
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--test_dir",
#     type=str,
#     required=True,
#     help="Test directory including the csv files"
# )
# parser.add_argument(
#     "--target_model",
#     type=str,
#     required=True,
#     help="Asteroid model to use"
# )
# parser.add_argument(
#     "--out_dir",
#     type=str,
#     default='ConvTasNet/eval/tmp',
#     required=True,
#     help="Directory where the eval results" " will be stored",
# )
# parser.add_argument(
#     "--use_gpu",
#     type=int,
#     default=0,
#     help="Whether to use the GPU for model execution"
# )
# parser.add_argument("--exp_dir",
#                     default="exp/tmp",
#                     help="Best serialized model path")
# parser.add_argument(
#     "--n_save_ex",
#     type=int,
#     default=10,
#     help="Number of audio examples to save, -1 means all"
# )
# parser.add_argument(
#     "--compute_wer",
#     type=int,
#     default=0,
#     help="Compute WER using ESPNet's pretrained model"
# )
#
# COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]
#
#
# def main(conf):
#     compute_metrics = COMPUTE_METRICS
#     wer_tracker = (
#         MockWERTracker()
#     )
#     model_path = os.path.join(conf["exp_dir"], "best_model.pth")
#     if conf["target_model"] == "UNet":
#         sys.path.append('UNet_model')
#         AsteroidModelModule = my_import("unet_model.UNet")
#     else:
#         sys.path.append('ConvTasNet_model')
#         AsteroidModelModule = my_import("conv_tasnet_norm.ConvTasNetNorm")
#     model = AsteroidModelModule.from_pretrained(model_path, sample_rate=conf["sample_rate"])
#     # model = ConvTasNet
#     # Handle device placement
#     if conf["use_gpu"]:
#         model.cuda()
#     test_set = PodcastMixDataloader(
#         csv_dir=conf["test_dir"],
#         sample_rate=conf["sample_rate"],
#         original_sample_rate=conf["original_sample_rate"],
#         segment=conf["segment"],
#         shuffle_tracks=False,
#         multi_speakers=conf["multi_speakers"]
#     )
#     # Used to reorder sources only
#
#     # Randomly choose the indexes of sentences to save.
#     eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
#     ex_save_dir = os.path.join(eval_save_dir, "examples/")
#     if conf["n_save_ex"] == -1:
#         conf["n_save_ex"] = len(test_set)
#     save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
#     # pdb.set_trace()
#     series_list = []
#
#     torch.no_grad().__enter__()
#     for idx in tqdm(range(len(test_set))):
#         # Forward the network on the mixture.
#         mix, sources = test_set[idx]
#
#         if conf["target_model"] == "UNet":
#             mix = mix.unsqueeze(0)
#         # get audio representations, pass the mix to the unet, it will normalize
#         # it, create the masks, pass them to audio, unnormalize them and return
#         est_sources = model(mix)
#
#         mix_np = mix.cpu().data.numpy()
#         if conf["target_model"] == "UNet":
#             mix_np = mix_np.squeeze(0)
#
#         sources_np = sources.cpu().data.numpy()
#         est_sources_np = est_sources.squeeze(0).cpu().data.numpy()
#
#         try:
#             utt_metrics = get_metrics(
#                 mix_np,
#                 sources_np,
#                 est_sources_np,
#                 sample_rate=conf["sample_rate"],
#                 metrics_list=COMPUTE_METRICS,
#                 average=False
#             )
#             series_list.append(pd.Series(utt_metrics))
#         except:
#             print("Error. Index", idx)
#             print(mix_np)
#             print(sources_np)
#             print(est_sources_np)
#
#         # Save some examples in a folder. Wav files and metrics as text.
#         if idx in save_idx:
#             local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx + 1))
#             os.makedirs(local_save_dir, exist_ok=True)
#             print(mix_np.shape)
#             sf.write(
#                 local_save_dir + "mixture.wav",
#                 mix_np,
#                 conf["sample_rate"]
#             )
#             # Loop over the sources and estimates
#             for src_idx, src in enumerate(sources_np):
#                 sf.write(
#                     local_save_dir + "s{}.wav".format(src_idx),
#                     src,
#                     conf["sample_rate"]
#                 )
#             for src_idx, est_src in enumerate(est_sources_np):
#                 # est_src *= np.max(np.abs(mix_np)) / np.max(np.abs(est_src))
#                 sf.write(
#                     local_save_dir + "s{}_estimate.wav".format(src_idx),
#                     est_src,
#                     conf["sample_rate"],
#                 )
#         # Write local metrics to the example folder.
#         with open(eval_save_dir + "metrics.json", "w") as f:
#             json.dump({k:v.tolist() for k,v in utt_metrics.items()}, f, indent=0)
#
#     # Save all metrics to the experiment folder.
#     all_metrics_df = pd.DataFrame(series_list)
#     all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))
#
#     # Print and save summary metrics
#     final_results = {}
#     for metric_name in compute_metrics:
#         input_metric_name = "input_" + metric_name
#         ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
#         final_results[metric_name] = all_metrics_df[metric_name].mean()
#         final_results[metric_name + "_imp"] = ldf.mean()
#
#     print("Overall metrics :")
#     print(final_results)
#     if conf["compute_wer"]:
#         print("\nWER report")
#         wer_card = wer_tracker.final_report_as_markdown()
#         print(wer_card)
#         # Save the report
#         with open(os.path.join(eval_save_dir, "final_wer.md"), "w") as f:
#             f.write(wer_card)
#
#     with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
#         json.dump({k:v.tolist() for k,v in final_results.items()}, f, indent=0)
#
#     # for publishing the model:
#     # model_dict = torch.load(model_path, map_location="cpu")
#     # os.makedirs(os.path.join(conf["exp_dir"], "publish_dir"), exist_ok=True)
#     # publishable = save_publishable(
#     #     os.path.join(conf["exp_dir"], "publish_dir"),
#     #     model_dict,
#     #     metrics=final_results,
#     #     train_conf=train_conf,
#     # )
#
#
# if __name__ == "__main__":
#     args = parser.parse_args()
#     arg_dic = dict(vars(args))
#     # Load training config
#     conf_path = os.path.join(args.exp_dir, "conf.yml")
#     with open(conf_path) as f:
#         train_conf = yaml.safe_load(f)
#     arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
#     arg_dic["segment"] = train_conf["data"]["segment"]
#     arg_dic["original_sample_rate"] = train_conf["data"]["original_sample_rate"]
#     arg_dic["multi_speakers"] = train_conf["training"]["multi_speakers"]
#     arg_dic["train_conf"] = train_conf
#
#     main(arg_dic)
