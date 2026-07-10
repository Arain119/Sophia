from __future__ import annotations

import time

import torch

from ml.core.engine.types import MetricsRow
from ml.core.engine.resume import build_train_state
from ml.core.engine.session import run_training_loop
from ml.core.engine.session_batches import scale_and_clip_grads, to_device_batch
from ml.core.engine.session_eval import supervised_eval
from ml.core.engine.session_loop import compute_throughput_metrics, should_fire_interval
from ml.tasks.sft.runtime_data import build_sft_data_state
from ml.tasks.sft.runtime_reports import (
    build_sft_checkpoint_report_callback,
    build_sft_finalize_fn,
)
from ml.training.model_contracts import require_compute_loss
from ml.training.posttrain.runtime import (
    prepare_posttrain_stage_runtime,
    save_stage_checkpoint,
)
from ml.training.posttrain.types import (
    PosttrainCheckpointSpec,
    PosttrainStageArgs,
    PosttrainStepResult,
)


def run(args: PosttrainStageArgs) -> None:
    resolved_max_seq_len, runtime = prepare_posttrain_stage_runtime(
        args=args,
        stage="sft",
        resolve_max_seq_len=lambda resolved_args: resolved_args.resolve_stage_max_seq_len(
            stage_kind="sft"
        ),
    )
    config = runtime.args.sft_stage_config(
        max_seq_len=int(resolved_max_seq_len),
    )
    data_state = build_sft_data_state(
        tokenizer=runtime.tokenizer,
        device=runtime.device,
        resume_state=dict(runtime.resume.train_state),
        config=config,
    )
    model = runtime.model
    optimizer = runtime.optimizer
    scheduler = runtime.scheduler
    ema = runtime.ema
    require_compute_loss(model, context="SFT training")
    finalize_outputs = build_sft_finalize_fn(
        runtime=runtime,
        model=model,
        tokenizer=runtime.tokenizer,
        config=config,
    )

    def _run_step(
        global_step: int,
        interval_wall_start: float,
        interval_train_time_s: float,
    ) -> PosttrainStepResult:
        step_train_start = time.time()
        model.train(True)
        optimizer.zero_grad(set_to_none=True)
        update_loss_sum = torch.zeros((), device=runtime.device, dtype=torch.float32)
        update_supervised_tokens = torch.zeros((), device=runtime.device, dtype=torch.int64)
        micro_batches = 0

        for _micro in range(int(config.accum_steps)):
            batch = next(data_state.train_iter)
            batch = to_device_batch(batch, device=runtime.device)
            model_inputs: dict[str, object] = dict(batch)
            model_inputs["use_cache"] = False
            model_inputs["compute_loss"] = True
            outputs = model(**model_inputs)
            if outputs.loss is None:
                raise RuntimeError("SFT model returned loss=None")
            labels = batch["labels"]
            supervised = (labels[:, 1:] != -100).sum(dtype=torch.int64)
            if int(supervised.item()) <= 0:
                continue
            loss_sum = outputs.loss * supervised.to(dtype=outputs.loss.dtype)
            loss_sum.backward()
            update_loss_sum.add_(loss_sum.detach().to(dtype=torch.float32))
            update_supervised_tokens.add_(supervised.detach().to(dtype=torch.int64))
            micro_batches += 1

        supervised_tokens = int(update_supervised_tokens.item())
        if supervised_tokens <= 0 or int(micro_batches) <= 0:
            raise RuntimeError("SFT update produced zero supervised tokens")

        grad_norm = scale_and_clip_grads(
            model=model,
            supervised_tokens=update_supervised_tokens,
            max_grad_norm=float(config.max_grad_norm),
        )
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update()

        step_train_elapsed = max(time.time() - step_train_start, 1e-6)
        data_state.samples_seen += int(config.batch_size) * int(micro_batches)
        data_state.running_loss_sum += float(update_loss_sum.item())
        data_state.running_tokens += int(supervised_tokens)

        should_log = should_fire_interval(
            global_step=int(global_step),
            interval=int(config.eval_interval),
            max_steps=int(config.max_steps),
        )
        should_save = should_fire_interval(
            global_step=int(global_step),
            interval=int(config.save_interval),
            max_steps=int(config.max_steps),
        )

        metrics_row: MetricsRow | None = None
        print_line: str | None = None
        reset_interval = False
        if should_log:
            avg_loss = float(data_state.running_loss_sum / max(data_state.running_tokens, 1))
            throughput = compute_throughput_metrics(
                tokens=int(data_state.running_tokens),
                train_time_s=float(interval_train_time_s) + float(step_train_elapsed),
                wall_time_s=max(time.time() - interval_wall_start, 1e-6),
            )
            metrics_row = {
                "stage": "sft",
                "step": int(global_step),
                "loss": avg_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "supervised_tokens": int(supervised_tokens),
                **throughput,
            }
            if grad_norm is not None:
                metrics_row["grad_norm"] = float(grad_norm.detach().item())
            if data_state.eval_iter is not None and int(config.eval_steps) > 0:
                eval_snapshot = data_state.eval_iter.state_dict()
                eval_loss = supervised_eval(
                    model=model,
                    data_iter=data_state.eval_iter,
                    steps=int(config.eval_steps),
                    base_dtype=runtime.base_dtype,
                )
                data_state.eval_iter.load_state_dict(eval_snapshot)
                metrics_row["eval_loss"] = float(eval_loss)
            if data_state.test_iter is not None and int(config.eval_steps) > 0:
                test_snapshot = data_state.test_iter.state_dict()
                test_loss = supervised_eval(
                    model=model,
                    data_iter=data_state.test_iter,
                    steps=int(config.eval_steps),
                    base_dtype=runtime.base_dtype,
                )
                data_state.test_iter.load_state_dict(test_snapshot)
                metrics_row["test_loss"] = float(test_loss)
            print_line = (
                f"[SFT] step={global_step} loss={avg_loss:.6f} lr={metrics_row['lr']:.3e} "
                f"toks/s={int(throughput['train_tokens_per_s'])} "
                f"wall_toks/s={int(throughput['wall_tokens_per_s'])}"
            )
            data_state.running_loss_sum = 0.0
            data_state.running_tokens = 0
            reset_interval = True

        train_state = (
            None
            if not should_save
            else build_train_state(
                stage="sft",
                iterators={
                    "train_iter_state": data_state.train_iter,
                    "eval_iter_state": data_state.eval_iter,
                    "test_iter_state": data_state.test_iter,
                },
                extra_state={
                    "running_loss_sum": float(data_state.running_loss_sum),
                    "running_tokens": int(data_state.running_tokens),
                    "samples_seen": int(data_state.samples_seen),
                },
            )
        )
        checkpoint_report_callback = (
            None
            if not should_save
            else build_sft_checkpoint_report_callback(
                runtime=runtime,
                model=model,
                tokenizer=runtime.tokenizer,
                config=config,
                global_step=int(global_step),
            )
        )
        return PosttrainStepResult(
            train_time_s=float(step_train_elapsed),
            metrics_row=metrics_row,
            print_line=print_line,
            checkpoint_train_state=train_state,
            checkpoint_report_callback=checkpoint_report_callback,
            reset_interval=reset_interval,
        )

    run_training_loop(
        output_dir=str(runtime.output_dir),
        start_step=int(runtime.start_step),
        max_steps=int(config.max_steps),
        step_fn=_run_step,
        finalize_fn=finalize_outputs,
        save_checkpoint_fn=lambda global_step, train_state: save_stage_checkpoint(
            checkpoint=PosttrainCheckpointSpec(
                output_dir=str(runtime.output_dir),
                step=int(global_step),
                args_payload=dict(runtime.run_args_payload),
                train_state=dict(train_state),
                save_total_limit=int(runtime.checkpoint_policy.save_total_limit),
                staging_dir=runtime.checkpoint_policy.staging_dir,
            ),
            model=runtime.model,
            optimizer=runtime.optimizer,
            scheduler=runtime.scheduler,
            ema=runtime.ema,
        ),
    )
