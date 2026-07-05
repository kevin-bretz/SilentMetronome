import argparse
import json
import os
from pathlib import Path

from audiotools import AudioSignal
from dac import DAC
import torch
from tqdm import tqdm

from stream_music_gen.constants import (
    DAC_SAMPLE_RATE,
    DAC_PRETRAINED_MODEL_PATH,
)
from stream_music_gen.dataset.token_dataset import get_audio_dataloader
from stream_music_gen.dataset.beat_grid_utils import (
    load_beat_grid,
    compute_beat_phase_frames,
    FRAME_RATE_HZ,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (10240, rlimit[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cocochorales", "moisesdb", "musdb", "slakh2100"],
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "valid", "test"],
        help="Split name",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        required=True,
        help="Maximum number of examples to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default="./audio_mixdown",
        help="Output directory",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers",
    )
    parser.add_argument(
        "--save_audio",
        action="store_true",
        help="Save audio",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index",
    )
    parser.add_argument(
        "--audio_duration",
        type=int,
        default=10,
        help="Audio Duration",
    )
    parser.add_argument(
        "--data_base_dir",
        type=str,
        default="stream_music_gen_data/causal_dac_codes_32khz",
        help="Base directory containing the token data",
    )
    parser.add_argument(
        "--rms_base_dir",
        type=str,
        default="stream_music_gen_data/rms_50hz",
        help="Base directory containing RMS data",
    )
    parser.add_argument(
        "--audio_base_dir",
        type=str,
        default="stream_music_gen_data/",
        help="Base directory containing audio files",
    )
    parser.add_argument(
        "--beat_grid_base_dir",
        type=str,
        default="stream_music_gen_data/beat_grids",
        help="Base directory containing per-track beat grid JSONs. "
        "Set to empty string to disable beat-phase preprocessing.",
    )
    parser.add_argument(
        "--skip_missing_beat_grid",
        action="store_true",
        default=True,
        help="If True (default), windows from tracks without a beat grid are "
        "skipped (num_example is not incremented).",
    )
    parser.add_argument(
        "--no_skip_missing_beat_grid",
        dest="skip_missing_beat_grid",
        action="store_false",
        help="Keep windows without beat grids; beat_cond.pt is not written.",
    )
    args = parser.parse_args()

    write_beat_cond = bool(args.beat_grid_base_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = DAC.load(DAC_PRETRAINED_MODEL_PATH).to(device)
    assert model.causal_decoder and model.causal_encoder
    model.eval()
    print(f"[info] Loaded DAC model from {DAC_PRETRAINED_MODEL_PATH}")

    batch_size = 64
    num_example = args.start_index
    pbar = tqdm(
        total=args.max_examples,
        desc=f"Extracting DAC codes for {args.dataset}, {args.split}",
        initial=args.start_index,
    )
    dataloader = get_audio_dataloader(
        batch_size=batch_size,
        num_workers=args.num_workers,
        split=args.split,
        dataset_names=[args.dataset],
        weights=[1],
        group_by_track="true",
        filter_stem_by_rms="true",
        data_base_dir=args.data_base_dir,
        rms_base_dir=args.rms_base_dir,
        audio_base_dir=args.audio_base_dir,
        target_sample_rate=DAC_SAMPLE_RATE,
        duration=args.audio_duration,
    )

    num_skipped_missing_grid = 0

    while True:
        for batch in dataloader:
            for i in range(len(batch["input_audio"])):
                # -- Beat-phase preprocessing: resolve track grid before any
                #    heavy lifting so we can skip missing-grid windows cheaply.
                beat_info = None
                track_name = None
                start_frame = None
                if write_beat_cond:
                    track_name = (
                        batch["track_name"][i]
                        if "track_name" in batch
                        else None
                    )
                    start_frame = (
                        int(batch["start_frame"][i])
                        if "start_frame" in batch
                        else None
                    )
                    grid = None
                    if track_name is not None:
                        grid = load_beat_grid(
                            dataset=args.dataset,
                            split=args.split,
                            track_name=track_name,
                            base_dir=args.beat_grid_base_dir,
                        )
                    if grid is None:
                        if args.skip_missing_beat_grid:
                            num_skipped_missing_grid += 1
                            continue  # skip without incrementing num_example
                        # else: proceed, beat_cond will not be written
                    else:
                        num_frames = int(
                            args.audio_duration * FRAME_RATE_HZ
                        )
                        beat_info = compute_beat_phase_frames(
                            grid,
                            start_frame=start_frame,
                            num_frames=num_frames,
                        )

                output_path = (
                    Path(args.output_dir)
                    / args.dataset
                    / args.split
                    / f"{num_example:07d}"
                )
                os.makedirs(output_path, exist_ok=True)
                for audio_type in ["input", "target"]:
                    audio = AudioSignal(
                        batch[f"{audio_type}_audio"][i],
                        sample_rate=DAC_SAMPLE_RATE,
                    ).to(device)

                    # NOTE(Shih-Lun): ensure no chunking
                    win_duration = audio.shape[-1] / DAC_SAMPLE_RATE + 1

                    with torch.no_grad():
                        encoder_outputs = model.compress(
                            audio, win_duration=win_duration
                        )
                        codes = encoder_outputs.codes.squeeze().cpu()

                    torch.save(codes, output_path / f"{audio_type}_codes.pt")
                    if args.save_audio:
                        torch.save(
                            audio, output_path / f"{audio_type}_audio.pt"
                        )

                # select only the metadata we need
                metadata = {
                    k: batch[k][i]
                    for k in [
                        "base_dir",
                        "input_inst_id",
                        "target_inst_id",
                        "input_token_path",
                        "target_token_path",
                        "input_audio_path",
                        "target_audio_path",
                    ]
                }
                # add num_stems to metadata
                metadata["num_stems"] = len(batch["input_inst_id"][i])

                # -- Beat-phase metadata + beat_cond.pt
                if write_beat_cond and beat_info is not None:
                    beat_cond_tensor = torch.from_numpy(
                        beat_info["beat_cond"]
                    ).float()
                    torch.save(
                        beat_cond_tensor, output_path / "beat_cond.pt"
                    )
                    metadata.update(
                        {
                            "track_name": track_name,
                            "start_frame": int(start_frame),
                            "start_time_sec": float(start_frame)
                            / float(FRAME_RATE_HZ),
                            "duration_frames": int(beat_cond_tensor.shape[0]),
                            "frame_rate_hz": int(FRAME_RATE_HZ),
                            "bpm_mean": float(beat_info["bpm_mean"]),
                            "bpm_log": float(beat_info["bpm_log"]),
                            "time_sig_num": int(beat_info["time_sig_num"]),
                            "time_sig_den": int(beat_info["time_sig_den"]),
                            "num_beats_in_window": int(
                                beat_info["num_beats_in_window"]
                            ),
                            "num_downbeats_in_window": int(
                                beat_info["num_downbeats_in_window"]
                            ),
                            "first_beat_frame_in_window": int(
                                beat_info["first_beat_frame_in_window"]
                            ),
                            "time_sig_change_in_window": bool(
                                beat_info["time_sig_change_in_window"]
                            ),
                            "tempo_change_in_window": bool(
                                beat_info["tempo_change_in_window"]
                            ),
                            "has_beat_grid": True,
                            "beat_frames_in_window": beat_info[
                                "beat_frames_in_window"
                            ],
                            "bar_positions_in_window": beat_info[
                                "bar_positions_in_window"
                            ],
                        }
                    )
                elif write_beat_cond:
                    # Track has no grid but we kept the window anyway.
                    metadata["has_beat_grid"] = False
                    if track_name is not None:
                        metadata["track_name"] = track_name
                    if start_frame is not None:
                        metadata["start_frame"] = int(start_frame)
                        metadata["start_time_sec"] = float(start_frame) / float(
                            FRAME_RATE_HZ
                        )

                with open(
                    output_path / "metadata.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(metadata, f)

                num_example += 1
                pbar.update(1)
                if num_example >= args.max_examples:
                    if write_beat_cond:
                        print(
                            f"[info] Skipped {num_skipped_missing_grid} "
                            f"windows with missing beat grids."
                        )
                    return


if __name__ == "__main__":
    main()
