"""Extract beat grids from Slakh2100 MIDI files.

For each track, parses all_src.mid to find tempo and time signature events,
computes beat positions in seconds, then converts to frame indices at 50 Hz.
Computes per-beat bar_position (beat index within the current bar).

Usage:
    python -m stream_music_gen.dataset.extract_beat_grid \
        --dataset slakh2100 --split train \
        --output_dir stream_music_gen_data/beat_grids

Output per track:
    {output_dir}/{dataset}/{split}/{TrackXXXXX}.json
    {
        "beat_frames": [0, 25, 50, ...],       # frame index of each beat at 50 Hz
        "beat_lengths": [25, 25, 25, ...],      # frames between consecutive beats
        "tempo_bpm": [120.0, 120.0, ...],       # BPM at each beat
        "bar_positions": [0, 1, 2, 3, 0, ...],  # beat index within bar (0 = downbeat)
        "time_sig_numerators": [4, 4, 4, ...],  # time sig numerator at each beat
        "time_signature": [4, 4],               # first time signature (numerator, denominator)
        "ticks_per_beat": 480
    }
"""

import argparse
import json
import os
from pathlib import Path

import mido

from stream_music_gen.constants import (
    DAC_FRAME_RATE_HZ,
    MIN_BEAT_LENGTH_FRAMES,
    DATASET_SPLITS,
)


def extract_beat_grid(midi_path: str, frame_rate: int = DAC_FRAME_RATE_HZ) -> dict:
    """Extract beat positions from a MIDI file.

    Walks through tempo and time signature events in track 0 to build
    a beat grid with proper handling of tempo and time signature changes.

    Args:
        midi_path: Path to the MIDI file.
        frame_rate: Frame rate in Hz for converting seconds to frames.

    Returns:
        Dict with beat_frames, beat_lengths, tempo_bpm, bar_positions,
        time_sig_numerators, time_signature, ticks_per_beat.
    """
    mid = mido.MidiFile(str(midi_path))
    ticks_per_beat = mid.ticks_per_beat

    # Collect tempo and time signature events from track 0
    tempo_events = []  # (tick, microseconds_per_beat)
    time_sig_events = []  # (tick, numerator, denominator)
    current_tick = 0

    for msg in mid.tracks[0]:
        current_tick += msg.time
        if msg.type == "set_tempo":
            tempo_events.append((current_tick, msg.tempo))
        elif msg.type == "time_signature":
            time_sig_events.append((current_tick, msg.numerator, msg.denominator))

    # Defaults if missing
    if not tempo_events:
        tempo_events = [(0, mido.bpm2tempo(120))]
    if not time_sig_events:
        time_sig_events = [(0, 4, 4)]

    # Sort by tick position
    tempo_events.sort(key=lambda x: x[0])
    time_sig_events.sort(key=lambda x: x[0])

    # First time signature (for backward compatibility)
    first_time_sig = (time_sig_events[0][1], time_sig_events[0][2])

    # Find the total duration in ticks (max tick across all tracks)
    max_tick = 0
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
        max_tick = max(max_tick, tick)

    # Build beat grid by walking through ticks
    beat_frames = []
    beat_bpms = []
    beat_bar_positions = []
    beat_ts_numerators = []

    # Current state
    tempo_idx = 0
    current_tempo = tempo_events[0][1]  # microseconds per beat
    current_time_sec = 0.0
    current_tick = 0

    # Time signature state
    ts_idx = 0
    current_numerator = time_sig_events[0][1]
    beats_since_ts_change = 0

    # Walk beat by beat
    while current_tick <= max_tick:
        # Check for time signature change at or before this tick
        while (
            ts_idx + 1 < len(time_sig_events)
            and time_sig_events[ts_idx + 1][0] <= current_tick
        ):
            ts_idx += 1
            current_numerator = time_sig_events[ts_idx][1]
            beats_since_ts_change = 0

        # Record this beat
        frame = round(current_time_sec * frame_rate)
        bpm = 60_000_000 / current_tempo
        bar_position = beats_since_ts_change % current_numerator

        beat_frames.append(frame)
        beat_bpms.append(round(bpm, 2))
        beat_bar_positions.append(bar_position)
        beat_ts_numerators.append(current_numerator)

        beats_since_ts_change += 1

        # Advance by one beat (ticks_per_beat)
        ticks_remaining = ticks_per_beat
        while ticks_remaining > 0:
            # Find next tempo change
            next_tempo_tick = None
            next_tempo_val = None
            if tempo_idx + 1 < len(tempo_events):
                next_tempo_tick = tempo_events[tempo_idx + 1][0]
                next_tempo_val = tempo_events[tempo_idx + 1][1]

            if next_tempo_tick is not None and next_tempo_tick < current_tick + ticks_remaining:
                # Tempo change within this beat
                ticks_before_change = next_tempo_tick - current_tick
                current_time_sec += mido.tick2second(
                    ticks_before_change, ticks_per_beat, current_tempo
                )
                ticks_remaining -= ticks_before_change
                current_tick = next_tempo_tick
                current_tempo = next_tempo_val
                tempo_idx += 1
            else:
                # No tempo change in remaining ticks
                current_time_sec += mido.tick2second(
                    ticks_remaining, ticks_per_beat, current_tempo
                )
                current_tick += ticks_remaining
                ticks_remaining = 0

    # Compute beat lengths (frames between consecutive beats)
    beat_lengths = []
    for i in range(len(beat_frames) - 1):
        length = beat_frames[i + 1] - beat_frames[i]
        beat_lengths.append(length)

    # Merge beats that are too short (< MIN_BEAT_LENGTH_FRAMES)
    # Keep the first beat's bar_position and time_sig for each merged group.
    merged_frames = [beat_frames[0]]
    merged_bpms = [beat_bpms[0]]
    merged_bar_positions = [beat_bar_positions[0]]
    merged_ts_numerators = [beat_ts_numerators[0]]
    accumulated = 0
    for i in range(len(beat_lengths)):
        accumulated += beat_lengths[i]
        if accumulated >= MIN_BEAT_LENGTH_FRAMES:
            merged_frames.append(beat_frames[i + 1])
            merged_bpms.append(beat_bpms[i + 1])
            merged_bar_positions.append(beat_bar_positions[i + 1])
            merged_ts_numerators.append(beat_ts_numerators[i + 1])
            accumulated = 0

    # Recompute lengths after merging
    merged_lengths = []
    for i in range(len(merged_frames) - 1):
        merged_lengths.append(merged_frames[i + 1] - merged_frames[i])

    return {
        "beat_frames": merged_frames,
        "beat_lengths": merged_lengths,
        "tempo_bpm": merged_bpms,
        "bar_positions": merged_bar_positions,
        "time_sig_numerators": merged_ts_numerators,
        "time_signature": list(first_time_sig),
        "ticks_per_beat": ticks_per_beat,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract beat grids from MIDI files")
    parser.add_argument(
        "--dataset", type=str, default="slakh2100", help="Dataset name"
    )
    parser.add_argument(
        "--split", type=str, required=True, choices=["train", "valid", "test"]
    )
    parser.add_argument(
        "--data_base_dir",
        type=str,
        default="stream_music_gen_data/slakh2100/original/slakh2100_redux_16k",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="stream_music_gen_data/beat_grids",
    )
    args = parser.parse_args()

    # Map split name (e.g., "valid" to "validation" for slakh2100)
    split_dir_name = DATASET_SPLITS.get(args.dataset, {}).get(args.split, args.split)

    data_dir = Path(args.data_base_dir) / split_dir_name
    output_dir = Path(args.output_dir) / args.dataset / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = sorted(data_dir.glob("Track*"))
    print(f"Found {len(tracks)} tracks in {data_dir}")

    success = 0
    errors = 0
    for track_dir in tracks:
        midi_path = track_dir / "all_src.mid"
        if not midi_path.exists():
            print(f"  SKIP {track_dir.name}: no all_src.mid")
            continue

        try:
            grid = extract_beat_grid(midi_path)
            output_path = output_dir / f"{track_dir.name}.json"
            with open(output_path, "w") as f:
                json.dump(grid, f)
            success += 1
        except Exception as e:
            print(f"  ERROR {track_dir.name}: {e}")
            errors += 1

    print(f"\nDone: {success} extracted, {errors} errors, {len(tracks)} total tracks")


if __name__ == "__main__":
    main()
