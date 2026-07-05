"""Training of encoder decoder model."""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Disable compile
# os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
torch._dynamo.config.disable = True

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import argbind

from functools import partial
from pathlib import Path

from stream_music_gen.base_trainer import Trainer
from stream_music_gen.lit_module.online_prefix_dec import (
    LitOnlinePrefixDecoderMultiOut,
)

GROUP = __file__
# Binding things only when this file is loaded
bind = partial(argbind.bind, group=GROUP)

Trainer = bind(Trainer, without_prefix=True)


@bind(without_prefix=True)
def main(args, save_dir: str = "", init_from_checkpoint: str = ""):

    # Create the lit module
    lit_module = LitOnlinePrefixDecoderMultiOut()  # Modified - Delay Pattern

    # Optional: warm-start model weights from a previous checkpoint.
    # Only `state_dict` is loaded (strict=False) so newly-added parameters
    # keep their init (e.g. zero-init beat-phase gate) and optimizer/LR/AMP
    # state start fresh. Don't use together with a save_dir that already has
    # its own ckpts --- once the run writes its first checkpoint, Lightning's
    # auto-resume via `load_from_latest_checkpoint` takes over and this flag
    # is ignored.
    if init_from_checkpoint:
        save_ckpt_files = list(Path(save_dir).glob("*.ckpt")) if save_dir else []
        if save_ckpt_files:
            print(
                f"[init_from_checkpoint] Skipped: save_dir already has "
                f"{len(save_ckpt_files)} ckpt(s); auto-resume will take over."
            )
        else:
            ckpt_path = Path(init_from_checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"init_from_checkpoint not found: {ckpt_path}"
                )
            print(f"[init_from_checkpoint] Loading weights from {ckpt_path}")
            raw = torch.load(
                str(ckpt_path), map_location="cpu", weights_only=False
            )
            state_dict = raw.get("state_dict", raw)

            # Reconcile the `_orig_mod.` prefix that torch.compile adds to
            # wrapped modules. Source or destination may or may not have it;
            # we add/strip to make them match.
            dest_sd = lit_module.state_dict()
            dest_has_orig = any("_orig_mod." in k for k in dest_sd.keys())
            src_has_orig = any("_orig_mod." in k for k in state_dict.keys())
            if dest_has_orig and not src_has_orig:
                # Insert `_orig_mod.` right after `model.` in source keys.
                state_dict = {
                    (
                        k.replace("model.", "model._orig_mod.", 1)
                        if k.startswith("model.")
                        and not k.startswith("model._orig_mod.")
                        else k
                    ): v
                    for k, v in state_dict.items()
                }
            elif src_has_orig and not dest_has_orig:
                state_dict = {
                    k.replace("model._orig_mod.", "model.", 1): v
                    for k, v in state_dict.items()
                }
            missing, unexpected = lit_module.load_state_dict(
                state_dict, strict=False
            )
            # Report cleanly so smoke tests can verify only new conditioner
            # parameters are missing.
            print(
                f"[init_from_checkpoint] missing_keys={len(missing)}, "
                f"unexpected_keys={len(unexpected)}"
            )
            if missing:
                print("  missing (first 20):")
                for k in missing[:20]:
                    print(f"    {k}")
            if unexpected:
                print("  unexpected (first 20):")
                for k in unexpected[:20]:
                    print(f"    {k}")

    # Get the dataloaders
    train_dataloader, val_dataloader = lit_module.get_dataloaders()

    # Train the model
    trainer = Trainer(
        args=args,
        lit_module=lit_module,
        save_dir=save_dir,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    trainer.train()


if __name__ == "__main__":
    args = argbind.parse_args(group=GROUP)
    argbind.dump_args(args, Path(args["save_dir"]) / "args.yml")
    with argbind.scope(args):
        main(args)
