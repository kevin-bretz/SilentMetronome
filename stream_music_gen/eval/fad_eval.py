import os
import glob
import sys

# Workaround, since there is some code that runs upon import
_old_argv = sys.argv
sys.argv = sys.argv[:1]
from frechet_audio_distance import FrechetAudioDistance

sys.argv = _old_argv


def calculate_fad(
    root_dir,
    gt_path: str = "ground_truth/pred_16000.wav",
    gen_path: str = "pred/pred_16000.wav",
    metric="vggish",
    verbose=False,
    normalize_rms_per_example=False,
    background_root_dirs=None,
    clap_submodel_name: str = "630k-audioset",
):
    """
    Calculate FAD score between reference and generated audio files matching
        specific patterns.

    Args:
        root_dir (str): Root directory containing folders of all examples
        gt_path (str): Relative path from single example folder to ground
            truth audio.
        gen_path (str): Relative path from single example folder to generated
            audio.
        metric (str): metric to use for embeddings
        verbose (bool): If we should print out outputs
        normalize_rms_per_example (bool): If we should match the rms between reference
            and generated audio.
        background_root_dirs (list[str], optional): if given, the reference
            (background) statistics are computed over gt_path files from ALL
            of these roots instead of root_dir only (e.g. pooled segment
            references for the LiveBand drift protocol).
        clap_submodel_name (str): LAION-CLAP checkpoint variant, only used
            when metric == "clap".
    """

    # Initialize FAD calculator
    if metric == "vggish":
        fad = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,  # This will resample to 16000 if not already
            use_pca=False,
            use_activation=False,
            verbose=verbose,
            audio_load_worker=1,
        )
    elif metric == "pann":
        fad = FrechetAudioDistance(
            model_name="pann",
            sample_rate=32000,
            use_pca=False,
            use_activation=False,
            verbose=verbose,
            audio_load_worker=1,
        )
    elif metric == "clap":
        fad = FrechetAudioDistance(
            model_name="clap",
            sample_rate=48000,  # package requirement; resamples internally
            submodel_name=clap_submodel_name,
            enable_fusion=False,
            verbose=verbose,
            audio_load_worker=1,
        )
    elif metric == "encodec":
        fad = FrechetAudioDistance(
            model_name="encodec", sample_rate=48000, channels=1, verbose=verbose
        )
    else:
        raise NotImplementedError(f"Unknown metric: {metric}")

    # Find files matching patterns
    if background_root_dirs:
        ref_files = sorted(
            f
            for d in background_root_dirs
            for f in glob.glob(os.path.join(d, "*", gt_path))
        )
    else:
        ref_files = sorted(glob.glob(os.path.join(root_dir, "*", gt_path)))
    gen_files = sorted(glob.glob(os.path.join(root_dir, "*", gen_path)))

    fad_score = fad.score(
        background_paths=ref_files,
        eval_paths=gen_files,
        dtype="float32",
        normalize_rms_per_example=normalize_rms_per_example,
    )
    return fad_score


if __name__ == "__main__":
    root_dir = "models/pref_dec_online_fv0_k50_beat_phase_dit_mp_cqt_aux_tt_future/model_predictions"
    gt_path = "ground_truth/pred.wav"
    gen_path = "pred/pred.wav"
    metric = "vggish"

    fad_score = calculate_fad(
        root_dir,
        gt_path=gt_path,
        gen_path=gen_path,
        metric=metric,
        verbose=True,
    )

    print(f"FAD Score: {fad_score}")
