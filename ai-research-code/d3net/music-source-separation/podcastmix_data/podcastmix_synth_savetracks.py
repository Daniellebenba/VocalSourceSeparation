import os
import pandas as pd
# import random

def load_all_mixtracks_synth():
    """Loads the specified dataset from commandline arguments
    Returns:
        train_dataset, validation_dataset
    """
    seed = 0
    root = '/Users/daniellebenbashat/Documents/IDC/signal_processing/FinalProject/data/podcastmix/podcastmix-synth/metadata'
    subs = ["train", "val", "test"]
    # sample_rate = 44100, original_sample_rate = 44100, segment = 6,  # 2,
    # download = False, samples_per_track = 64, source_augmentations = lambda audio: audio,
    # shuffle_tracks = False, multi_speakers = False,
    # random_track_mix = False, dtype = np.float32, seed = 42, rng = None,  # from origin MusDB
    # to_stereo: bool = True
    for sub in subs:

        csv_dir = os.path.join(root, sub)
        # declare dataframes
        speech_csv_path = os.path.join(csv_dir, 'speech.csv')
        music_csv_path = os.path.join(csv_dir, 'music.csv')

        df_speech = pd.read_csv(speech_csv_path, engine='python')
        df_music = pd.read_csv(music_csv_path, engine='python')

        n = min([len(df_speech), len(df_music)])
        df_speech = df_speech.sample(n, ignore_index=True, random_state=seed)
        df_music = df_music.sample(n, ignore_index=True, random_state=seed)

        # initialize indexes
        speech_inxs = list(range(len(df_speech)))
        music_inxs = list(range(len(df_music)))

        # random.shuffle(music_inxs)
        # random.shuffle(speech_inxs)

        df_speech["index_merge"], df_music["index_merge"] = speech_inxs, music_inxs
        df = pd.concat([df_speech, df_music], axis=1)

        mix_csv_path = os.path.join(csv_dir, f'{sub}.csv')
        print(f"save mix speech music in {mix_csv_path}")
        df.to_csv(mix_csv_path)


if __name__ == "__main__":
    load_all_mixtracks_synth()
