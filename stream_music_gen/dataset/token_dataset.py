"""Dataset and Dataloader for audio tokens."""

from pathlib import Path
import pandas as pd
from typing import Callable, Optional, List
import json

from torch.utils.data import Dataset
import torch
import numpy as np

from transformers import EncodecModel
from dac import DAC

from stream_music_gen.constants import inst_name_to_inst_class_id
from stream_music_gen.dataset.transforms import (
    ComposeTransform,
    SelectStems,
    CodecTokensToLMTokens,
    Serialize,
    AddSpecialTokens,
    PadTokens,
    RemoveKeys,
    LoadRMS,
    SubmixInputEmbEncodec,
    SubmixInputEmbDAC,
    PrecomputedInputTokenToEmbDAC,
    PadEmbs,
)

from stream_music_gen.constants import (
    MIDI_CATEGORIES,
    RMS_FRAME_RATE_HZ,
    ENCODEC_FRAME_RATE_HZ,
    ENCODEC_NUM_CODEBOOK_PER_LAYER,
    ENCODEC_SAMPLE_RATE,
    DAC_FRAME_RATE_HZ,
    DAC_NUM_CODEBOOK_PER_LAYER,
    DAC_SAMPLE_RATE,
    DAC_PRETRAINED_MODEL_PATH,
    DATASET_SPLITS,
)


class TokenDataset(Dataset):
    """Dataset for loading audio tokens."""

    def __init__(
        self,
        base_dir: str,
        dataset_names: List[str],
        split: str,
        weights: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        group_by_track: bool = True,
        audio_base_dir: Optional[str] = None,
    ):
        if weights is not None and len(weights) != len(dataset_names):
            raise ValueError(
                "Number of weights must be equal to the number of dataset names."
            )
        if weights is not None:
            if not all(isinstance(w, int) for w in weights):
                raise ValueError("All weights must be integers.")
            if not all(w > 0 for w in weights):
                raise ValueError("All weights must be greater than 0.")
        if transform is None:
            raise ValueError("Transform must be provided.")

        self.base_dir = Path(base_dir)
        self.dataset_names = dataset_names
        self.split = split
        # tokenizer-specific transform is also included here
        self.transform = transform
        if weights is None:
            weights = [1] * len(dataset_names)
        self.weights = weights
        self.group_by_track = group_by_track
        self.audio_base_dir = audio_base_dir
        self.load_metadata()

    def load_metadata(self):
        metadata_all = []
        for i in range(len(self.dataset_names)):
            dataset_name = self.dataset_names[i]
            split = DATASET_SPLITS[dataset_name][self.split]
            metadata_file = (
                self.base_dir / dataset_name / f"{split}_metadata.parquet"
            )
            metadata = pd.read_parquet(metadata_file)
            if self.group_by_track:
                # Group metadata by track
                metadata = (
                    metadata.groupby("track_name").agg(list).reset_index()
                )
            metadata["dataset_name"] = dataset_name
            weight = self.weights[i]
            metadata_all.extend([metadata] * weight)
            print(
                f"{dataset_name}-{split}: {len(metadata)} items, "
                f"weight={weight}, total={len(metadata) * weight}"
            )
        metadata_all = pd.concat(metadata_all, ignore_index=True)
        self.metadata = metadata_all
        print(f"Loaded all metadata for {len(self.metadata)} items.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx].to_dict()
        row["base_dir"] = self.base_dir
        dataset_base_dir = self.base_dir / row["dataset_name"]
        if self.group_by_track:
            row["audio_token_path"] = [
                str(dataset_base_dir / Path(p)) for p in row["audio_token_path"]
            ]
        else:
            row["audio_token_path"] = str(
                dataset_base_dir / Path(row["audio_token_path"])
            )

        item = self.transform(row)
        return item


class PrecomputedTokenDataset(Dataset):
    """Dataset for loading precomputed audio tokens."""

    def __init__(
        self,
        base_dir: str,
        dataset_names: List[str],
        split: str,
        weights: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        load_audio: bool = False,
        num_rvq_layers: int = 4,
        max_frame_length: int = 500,
        duration: float = 10.0,
        sample_rate: int = 32000,
    ):
        """Dataset for loading precomputed audio tokens.

        Args:
            base_dir (str): Base directory containing the token data.
            dataset_names (List[str]): List of dataset names to load.
            split (str): Dataset split to load.
            weights (Optional[List[int]], optional): Weights for each dataset in dataset_names. Defaults to None.
            transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
            load_audio (bool, optional): Whether to load raw audio data. Defaults to False.
            num_rvq_layers (int, optional): Number of RVQ layers. Defaults to 4.
            max_frame_length (int, optional): Maximum frame length. Defaults to 500.
            duration (float, optional): Duration of audio segments in seconds. Defaults to 10.0.
            sample_rate (int, optional): Sample rate of the audio. Defaults to 32000.
        Raises:
            ValueError: If the split is "train" and load_audio is True.
        """
        self.base_dir = Path(base_dir)
        self.dataset_names = dataset_names
        self.split = split
        self.weights = weights
        self.transform = transform
        self.load_metadata()
        self.load_audio = load_audio
        if load_audio and split == "train":
            raise ValueError("Cannot load audio for train split.")
        self.num_rvq_layers = num_rvq_layers
        self.max_frame_length = max_frame_length
        self.duration = duration
        self.sample_rate = sample_rate

    def load_metadata(self):
        metadata_all = []
        for i in range(len(self.dataset_names)):
            dataset_name = self.dataset_names[i]
            dataset_base_dir = self.base_dir / dataset_name / self.split
            # find all folders in the dataset_base_dir
            metadata_single_dataset = sorted(list(dataset_base_dir.glob("*/")))
            metadata_all.extend(metadata_single_dataset * self.weights[i])
            print(
                f"Loaded {len(metadata_single_dataset)} items for "
                f"{dataset_name}-{self.split}, with weight {self.weights[i]}, "
                f"at {dataset_base_dir}"
            )
        self.metadata = metadata_all
        print(f"Loaded all metadata for {len(self.metadata)} items.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data_dir = self.metadata[idx]
        # print(data_dir)

        input_token_path = data_dir / "input_codes.pt"
        target_token_path = data_dir / "target_codes.pt"
        input_token = torch.load(input_token_path, weights_only=True)
        target_token = torch.load(target_token_path, weights_only=True)

        if (
            input_token.shape[1] < self.max_frame_length
            or target_token.shape[1] < self.max_frame_length
        ):
            raise ValueError(
                f"Input token length ({input_token.shape[1]}) or target token length "
                f"({target_token.shape[1]}) is less than max frame length "
                f"({self.max_frame_length})"
            )
        else:
            input_token = input_token[
                : self.num_rvq_layers, : self.max_frame_length
            ]
            target_token = target_token[
                : self.num_rvq_layers, : self.max_frame_length
            ]

        metadata_path = data_dir / "metadata.json"
        metadata = json.load(open(metadata_path))

        # We don't need to mask the tokens here, because in the dataset dump,
        # the audio are padded with silence, so the tokens are all valid.
        length = input_token.shape[1]
        item = {
            "input_token": input_token,
            "target_token": target_token,
            "input_token_mask": torch.ones(length, dtype=torch.bool),
            "target_token_mask": torch.ones(length, dtype=torch.bool),
            **metadata,
        }

        # Beat-phase conditioning, optional. Only present in the
        # `precompute_audio_mixdown_20s_beat` data root.
        beat_cond_path = data_dir / "beat_cond.pt"
        if beat_cond_path.exists():
            beat_cond = torch.load(beat_cond_path, weights_only=True).float()
            # Align with max_frame_length, same as tokens above.
            beat_cond = beat_cond[: self.max_frame_length]
            T = beat_cond.shape[0]
            item["beat_cond"] = beat_cond
            item["bpm_log"] = torch.tensor(
                float(metadata.get("bpm_log", 0.0)), dtype=torch.float32
            )
            item["time_sig_num"] = torch.tensor(
                int(metadata.get("time_sig_num", 4)), dtype=torch.long
            )
            item["time_sig_den"] = torch.tensor(
                int(metadata.get("time_sig_den", 4)), dtype=torch.long
            )
            item["time_sig_change"] = torch.tensor(
                1.0 if bool(metadata.get("time_sig_change_in_window", False)) else 0.0,
                dtype=torch.float32,
            )
            item["tempo_change"] = torch.tensor(
                1.0 if bool(metadata.get("tempo_change_in_window", False)) else 0.0,
                dtype=torch.float32,
            )

            # Per-frame local BPM. Computed from beat_frames_in_window (frame
            # indices within the window where beats occur). Between consecutive
            # beats, local_BPM = 60 / interval_sec where interval_sec =
            # (next_beat_frame - prev_beat_frame) / frame_rate_hz. Frames
            # before the first beat use the first interval and frames after
            # the last beat use the last interval. Falls back to bpm_mean
            # for the whole window if beat_frames are missing.
            beat_frames = metadata.get("beat_frames_in_window", None)
            frame_rate = float(metadata.get("frame_rate_hz", 50))
            bpm_mean_for_log = float(metadata.get("bpm_mean", 120.0))
            if beat_frames and len(beat_frames) >= 2 and frame_rate > 0:
                bf = np.asarray(beat_frames, dtype=np.float64)
                # interval (in frames) at each beat boundary, forward diff
                intervals = np.diff(bf)
                # local BPM at beat i = 60 / (interval[i] / frame_rate)
                bpm_at_beat = 60.0 * frame_rate / np.clip(intervals, 1e-3, None)
                # Map each frame to its bracketing beat interval, giving
                # piecewise-constant local BPM. Frames before bf[0] take
                # bpm_at_beat[0], frames in [bf[i], bf[i+1]) take
                # bpm_at_beat[i], frames at/after bf[-1] take bpm_at_beat[-1].
                local_bpm = np.full(T, bpm_at_beat[-1], dtype=np.float64)
                # Frames before first beat
                first_bf = int(np.floor(bf[0]))
                local_bpm[: max(0, min(first_bf, T))] = bpm_at_beat[0]
                # Frames inside intervals
                for i in range(len(intervals)):
                    lo = int(np.floor(bf[i]))
                    hi = int(np.floor(bf[i + 1]))
                    lo = max(0, min(lo, T))
                    hi = max(0, min(hi, T))
                    if hi > lo:
                        local_bpm[lo:hi] = bpm_at_beat[i]
                local_bpm_log = np.log(np.clip(local_bpm, 1e-3, None) / 120.0)
            else:
                # Fallback: broadcast the per-window mean across all frames.
                local_bpm_log = np.full(
                    T, np.log(max(bpm_mean_for_log, 1e-3) / 120.0)
                )
            item["local_bpm_log"] = torch.from_numpy(local_bpm_log).float()

        # Multipitch (presence + velocity) and CQT (log-magnitude). All
        # optional, present only after running extract_target_multipitch.py
        # / extract_target_cqt.py against the precomputed window root.
        target_multipitch_path = data_dir / "target_multipitch.pt"
        if target_multipitch_path.exists():
            mp = torch.load(
                target_multipitch_path, weights_only=True
            ).float()[: self.max_frame_length]
            item["target_multipitch"] = mp
            has_mp_path = data_dir / "target_has_multipitch.pt"
            has_mp = (
                bool(torch.load(has_mp_path, weights_only=True))
                if has_mp_path.exists()
                else True
            )
            item["target_has_multipitch"] = torch.tensor(
                1.0 if has_mp else 0.0, dtype=torch.float32
            )
            target_velocity_path = data_dir / "target_velocity.pt"
            if target_velocity_path.exists():
                vel = torch.load(
                    target_velocity_path, weights_only=True
                ).float()[: self.max_frame_length]
                item["target_velocity"] = vel

        target_cqt_path = data_dir / "target_cqt.pt"
        if target_cqt_path.exists():
            cqt = torch.load(
                target_cqt_path, weights_only=True
            ).float()[: self.max_frame_length]
            item["target_cqt"] = cqt

        if self.load_audio:
            input_audio_path = data_dir / "input_audio.pt"
            target_audio_path = data_dir / "target_audio.pt"
            # load audio as AudioSignal
            input_audio = torch.load(input_audio_path, map_location="cpu")
            target_audio = torch.load(target_audio_path, map_location="cpu")
            item["input_audio"] = input_audio.audio_data.squeeze()
            item["target_audio"] = target_audio.audio_data.squeeze()
            # Crop the audio to the duration
            item["input_audio"] = item["input_audio"][
                : int(self.duration * self.sample_rate)
            ]
            item["target_audio"] = item["target_audio"][
                : int(self.duration * self.sample_rate)
            ]

        if self.transform is not None:
            item = self.transform(item)

        return item


def get_dataloader(
    dataset_names: List[str] = ["slakh2100"],
    data_base_dir: str = "stream_music_gen_data/causal_dac_codes_32khz",
    rms_base_dir: str = "stream_music_gen_data/rms_50hz",
    audio_base_dir: str = "stream_music_gen_data/",
    load_audio: str = "false",
    max_num_input_stems: Optional[int] = None,
    split: str = "train",
    weights: Optional[List[int]] = None,
    batch_size: int = 1,
    num_workers: int = 8,
    num_rvq_layers: int = 4,
    duration: int = 10,
    codec_name: str = "causal_dac_32khz",
    group_by_track: str = "true",
    filter_stem_by_rms: str = "true",
    shuffle: bool = True,
    pattern: str = "flatten",  # "flatten" or "multilayer"
    add_inst_tokens: bool = False,
    copy_target_to_input: bool = False,
) -> List[torch.utils.data.DataLoader]:
    """Create a DataLoader for audio token datasets.

    Args:
        dataset_names: List of dataset names to load.
        data_base_dir: Base directory containing the token data.
        rms_base_dir: Base directory containing RMS data.
        audio_base_dir: Base directory containing audio files.
        load_audio: Whether to load raw audio data.
        max_num_input_stems: Maximum number of input stems to load.
        split: Dataset split to load.
        weights: Weights for each dataset in dataset_names.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of worker processes for data loading.
        num_rvq_layers: Number of RVQ layers.
        duration: Duration of audio segments in seconds.
        codec_name: Name of the codec to use ("encodec_32khz" or "causal_dac_32khz").
        group_by_track: Whether to group data by track.
        filter_stem_by_rms: Whether to filter stems based on RMS values.
        shuffle: Whether to shuffle the data.
        pattern: the pattern to use for the output tokens
            - "flatten": flatten the tokens
            - "multilayer": use multilayer tokens (for musicgendelay-patterning)
        add_inst_tokens: Whether to add instrument IDs as conditioning at the beginning of sequences.
        copy_target_to_input: Whether to use the target tokens as input tokens.

    Returns:
        torch.utils.data.DataLoader: A DataLoader instance configured for loading audio token data.

    Raises:
        ValueError: If the codec_name is not supported or if weights are provided but don't match the number of datasets.
    """

    # To compatible with argbind
    group_by_track = group_by_track.lower() == "true"
    filter_stem_by_rms = filter_stem_by_rms.lower() == "true"
    load_audio = load_audio.lower() == "true"

    num_instruments = len(MIDI_CATEGORIES)

    if codec_name == "encodec_32khz":
        codebook = EncodecModel.from_pretrained(
            "facebook/encodec_32khz"
        ).quantizer
        sample_rate = ENCODEC_SAMPLE_RATE
        num_codebook_per_layer = ENCODEC_NUM_CODEBOOK_PER_LAYER
        frame_rate_hz = ENCODEC_FRAME_RATE_HZ
        submix_transform = SubmixInputEmbEncodec(
            codebook=codebook,
        )
    elif codec_name == "causal_dac_32khz":
        codebook = DAC.load(DAC_PRETRAINED_MODEL_PATH).quantizer
        sample_rate = DAC_SAMPLE_RATE
        num_codebook_per_layer = DAC_NUM_CODEBOOK_PER_LAYER
        frame_rate_hz = DAC_FRAME_RATE_HZ
        submix_transform = SubmixInputEmbDAC(
            codebook=codebook,
        )
    else:
        raise ValueError(f"Unknown codec name: {codec_name}")

    # Support multi-layered dataset, where items are [num_rvq_layers, T] (not flattened)
    max_frame_length = duration * frame_rate_hz
    if pattern == "multilayer":
        max_token_length = max_frame_length
    else:
        max_token_length = duration * frame_rate_hz * num_rvq_layers

    transforms = [
        LoadRMS(
            rms_base_dir=rms_base_dir,
        ),
        SelectStems(
            ignore_instrument_ids=[
                inst_name_to_inst_class_id("unknown"),
                inst_name_to_inst_class_id("vocal"),
            ],
            num_rvq_layers=num_rvq_layers,
            load_audio=load_audio,
            audio_base_dir=audio_base_dir,
            duration=10,
            target_sample_rate=sample_rate,
            token_crop_length=max_frame_length,
            max_num_input_stems=max_num_input_stems,
            rms_threshold=-60.0,
            num_retries=16,
            min_overlap_portion=0.5,
            filter_stem_by_rms=filter_stem_by_rms,
            rms_token_hz=RMS_FRAME_RATE_HZ,
            copy_target_to_input=copy_target_to_input,
        ),
        submix_transform,
    ]

    if pattern == "multilayer":
        transforms.extend(
            [
                CodecTokensToLMTokens(
                    num_special_tokens=1,
                    num_instrument_tokens=num_instruments,
                    num_codebook_per_layer=num_codebook_per_layer,
                    shared=True,  # All rvq levels have same range of ids.
                ),
                AddSpecialTokens(
                    multilayer=True,
                    add_bos=False,
                    add_eos=False,
                    add_inst_tokens=add_inst_tokens,
                ),
                PadEmbs(
                    key="input_emb",
                    input_length=max_frame_length,  # No divide by n_rvq
                ),
                # Padding is different for multilayer.
                PadTokens(
                    key="target_token",
                    input_length=max_token_length
                    + add_inst_tokens,  # 1 for inst token
                    multilayer=True,
                ),
            ]
        )
    elif pattern == "flatten":
        transforms.extend(
            [
                CodecTokensToLMTokens(
                    num_special_tokens=3,
                    num_instrument_tokens=num_instruments,
                    num_codebook_per_layer=num_codebook_per_layer,
                ),
                Serialize(),
                AddSpecialTokens(add_inst_tokens=add_inst_tokens),
                PadEmbs(key="input_emb", input_length=max_frame_length),
                PadTokens(
                    key="target_token", input_length=max_token_length + 2
                ),
            ]
        )
    if load_audio:
        transforms.extend(
            [
                PadTokens(
                    key="target_audio",
                    input_length=duration * sample_rate,
                ),
                PadTokens(
                    key="input_audio",
                    input_length=duration * sample_rate,
                ),
            ]
        )

    dataset = TokenDataset(
        base_dir=data_base_dir,
        dataset_names=dataset_names,
        split=split,
        weights=weights,
        group_by_track=group_by_track,
        transform=ComposeTransform(transforms),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=default_collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return dataloader


def get_audio_dataloader(
    dataset_names: List[str] = ["slakh2100"],
    data_base_dir: str = "stream_music_gen_data/causal_dac_codes_32khz",
    rms_base_dir: str = "stream_music_gen_data/rms_50hz",
    audio_base_dir: str = "stream_music_gen_data/",
    target_sample_rate: int = 32000,
    max_num_input_stems: Optional[int] = None,
    split: str = "train",
    weights: Optional[List[int]] = None,
    batch_size: int = 1,
    num_workers: int = 8,
    duration: int = 10,
    group_by_track: str = "true",
    filter_stem_by_rms: str = "true",
    shuffle: bool = True,
    copy_target_to_input: bool = False,
) -> List[torch.utils.data.DataLoader]:
    """Create a DataLoader for loading raw audio data.

    Args:
        dataset_names: List of dataset names to load.
        data_base_dir: Base directory containing the token data.
        rms_base_dir: Base directory containing RMS data.
        audio_base_dir: Base directory containing audio files.
        target_sample_rate: Target sample rate for the audio data.
        max_num_input_stems: Maximum number of input stems to load.
        split: Dataset split to load ("train", "val", or "test").
        weights: Weights for each dataset in dataset_names.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of worker processes for data loading.
        duration: Duration of audio segments in seconds.
        group_by_track: Whether to group data by track ("true" or "false").
        filter_stem_by_rms: Whether to filter stems based on RMS values ("true" or "false").
        shuffle: Whether to shuffle the data.
        copy_target_to_input: Whether to use the target audio as input audio.

    Returns:
        torch.utils.data.DataLoader: A DataLoader instance configured for loading raw audio data.
    """

    # To compatible with argbind
    group_by_track = group_by_track.lower() == "true"
    filter_stem_by_rms = filter_stem_by_rms.lower() == "true"
    load_audio = True

    # Support multi-layered dataset, where items are [num_rvq_layers, T] (not flattened)
    frame_rate_hz = 50  # for rms
    max_frame_length = duration * frame_rate_hz
    sample_rate = target_sample_rate

    transforms = [
        LoadRMS(
            rms_base_dir=rms_base_dir,
        ),
        SelectStems(
            ignore_instrument_ids=[
                inst_name_to_inst_class_id("unknown"),
                inst_name_to_inst_class_id("vocal"),
            ],
            num_rvq_layers=4,  # no effect
            load_audio=load_audio,
            audio_base_dir=audio_base_dir,
            duration=10,
            target_sample_rate=sample_rate,
            token_crop_length=max_frame_length,
            max_num_input_stems=max_num_input_stems,
            rms_threshold=-60.0,
            num_retries=16,
            min_overlap_portion=0.5,
            filter_stem_by_rms=filter_stem_by_rms,
            rms_token_hz=RMS_FRAME_RATE_HZ,
            copy_target_to_input=copy_target_to_input,
            load_audio_tokens=False,
        ),
    ]

    transforms.extend(
        [
            PadTokens(
                key="target_audio",
                input_length=duration * sample_rate,
            ),
            PadTokens(
                key="input_audio",
                input_length=duration * sample_rate,
            ),
        ]
    )

    dataset = TokenDataset(
        base_dir=data_base_dir,
        dataset_names=dataset_names,
        split=split,
        weights=weights,
        group_by_track=group_by_track,
        transform=ComposeTransform(transforms),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=default_collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return dataloader


def get_precomputed_token_dataloader(
    dataset_names: List[str] = ["slakh2100"],
    data_base_dir: str = "stream_music_gen_data/causal_dac_codes_32khz",
    split: str = "train",
    weights: Optional[List[int]] = None,
    batch_size: int = 1,
    num_workers: int = 8,
    num_rvq_layers: int = 4,
    duration: float = 10,
    shuffle: bool = True,
    load_audio: str = "false",
    pattern: str = "flatten",
    add_inst_tokens: bool = False,
    **kwargs,  # compatible with get_dataloader
) -> List[torch.utils.data.DataLoader]:
    """Create a DataLoader for loading raw audio data.

    Args:
        dataset_names: List of dataset names to load.
        data_base_dir: Base directory containing the token data.
        rms_base_dir: Base directory containing RMS data.
        audio_base_dir: Base directory containing audio files.
        target_sample_rate: Target sample rate for the audio data.
        max_num_input_stems: Maximum number of input stems to load.
        split: Dataset split to load ("train", "val", or "test").
        weights: Weights for each dataset in dataset_names.
        batch_size: Batch size for the DataLoader.
        num_workers: Number of worker processes for data loading.
        num_rvq_layers: Number of RVQ layers.
        duration: Duration of audio segments in seconds.
        shuffle: Whether to shuffle the data.
        load_audio: Whether to load audio data.
        pattern: the pattern to use for the output tokens
            - "flatten": flatten the tokens
            - "multilayer": use multilayer tokens (for musicgendelay-patterning)
        add_inst_tokens: Whether to add instrument IDs as conditioning at the beginning of sequences.

    Returns:
        torch.utils.data.DataLoader: A DataLoader instance configured for loading raw audio data.
    """

    # To compatible with argbind
    load_audio = load_audio.lower() == "true"

    num_instruments = len(MIDI_CATEGORIES)

    # hard-code to use DAC
    codebook = DAC.load(DAC_PRETRAINED_MODEL_PATH).quantizer
    num_codebook_per_layer = DAC_NUM_CODEBOOK_PER_LAYER
    frame_rate_hz = DAC_FRAME_RATE_HZ
    submix_transform = PrecomputedInputTokenToEmbDAC(
        codebook=codebook,
    )
    sample_rate = DAC_SAMPLE_RATE

    max_frame_length = int(duration * frame_rate_hz)

    transforms = [submix_transform]

    if pattern == "multilayer":
        transforms.extend(
            [
                CodecTokensToLMTokens(
                    num_special_tokens=1,
                    num_instrument_tokens=num_instruments,
                    num_codebook_per_layer=num_codebook_per_layer,
                    shared=True,  # All rvq levels have same range of ids.
                ),
                AddSpecialTokens(
                    multilayer=True,
                    add_bos=False,
                    add_eos=False,
                    add_inst_tokens=add_inst_tokens,
                ),
            ]
        )
    elif pattern == "flatten":
        transforms.extend(
            [
                CodecTokensToLMTokens(
                    num_special_tokens=3,
                    num_instrument_tokens=num_instruments,
                    num_codebook_per_layer=num_codebook_per_layer,
                ),
                Serialize(),
                AddSpecialTokens(add_inst_tokens=add_inst_tokens),
            ]
        )
    if weights is None:
        weights = [1] * len(dataset_names)

    dataset = PrecomputedTokenDataset(
        base_dir=data_base_dir,
        dataset_names=dataset_names,
        split=split,
        weights=weights,
        transform=ComposeTransform(transforms),
        load_audio=load_audio,
        num_rvq_layers=num_rvq_layers,
        max_frame_length=max_frame_length,
        duration=duration,
        sample_rate=sample_rate,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=default_collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return dataloader


def default_collate_fn(batch):
    """Collate function for audio token dataset."""
    keys = batch[0].keys()
    collated_batch = {}
    for key in keys:
        if not all(key in item for item in batch):
            continue
        if isinstance(batch[0][key], torch.Tensor):
            collated_batch[key] = torch.stack([item[key] for item in batch])
        else:
            collated_batch[key] = [item[key] for item in batch]
    return collated_batch


if __name__ == "__main__":
    from tqdm import tqdm

    dataloader = get_dataloader(
        batch_size=8,
        num_workers=4,
        split="train",
        dataset_names=[
            "slakh2100",
            "cocochorales",
            "moisesdb",
        ],  # ["slakh2100", "cocochorales", "moisesdb"],
        weights=[18, 1, 100],
        group_by_track="true",
        filter_stem_by_rms="true",
        data_base_dir="stream_music_gen_data/causal_dac_codes_32khz",
        rms_base_dir="stream_music_gen_data/rms_50hz",
        pattern="multilayer",
    )
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        pass
        # print(batch)
        # if i > 0:
        #     break
