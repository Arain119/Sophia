from __future__ import annotations

import torch

from ml.core.engine.resume import (
    read_resume_state_value,
    restore_iterator_states,
)
from ml.core.engine.session_batches import maybe_prefetch_batch_iterator
from ml.core.engine.types import StatePayload
from ml.tasks.sft.spec import SftDataState, SftStageConfig
from ml.training.posttrain.contracts import SupportsPosttrainTokenizer
from ml.training.posttrain.data import (
    ChatExample,
    SupervisedBatchIterator,
    encode_supervised_example,
    load_supervised_examples,
)


def _select_train_examples(
    *,
    dataset: list[ChatExample],
    tokenizer: SupportsPosttrainTokenizer,
    config: SftStageConfig,
) -> list[ChatExample]:
    selection = str(config.selection or "all_examples").strip()
    if selection == "all_examples":
        return list(dataset)
    if selection != "long_examples_only":
        raise ValueError(f"unsupported SFT selection policy: {selection!r}")
    selected: list[ChatExample] = []
    for example in dataset:
        encoded = encode_supervised_example(
            tokenizer=tokenizer,
            example=example,
            max_seq_len=0,
            crop_policy="tail_tokens",
        )
        if int(encoded.input_ids.numel()) > int(config.max_seq_len):
            selected.append(example)
    if not selected:
        raise ValueError(
            "SFT selection policy produced an empty training set: "
            f"selection={selection!r} max_seq_len={int(config.max_seq_len)}"
    )
    return selected


def build_sft_data_state(
    *,
    tokenizer: SupportsPosttrainTokenizer,
    device: torch.device,
    resume_state: StatePayload,
    config: SftStageConfig,
) -> SftDataState:
    train_examples = load_supervised_examples(str(config.train_data_path))
    train_examples = _select_train_examples(
        dataset=train_examples,
        tokenizer=tokenizer,
        config=config,
    )
    eval_examples = (
        load_supervised_examples(str(config.eval_data_path))
        if str(config.eval_data_path).strip()
        else None
    )
    test_examples = (
        load_supervised_examples(str(config.test_data_path))
        if str(config.test_data_path).strip()
        else None
    )

    print(
        "[INFO] SFT runtime config | "
        f"max_seq_len={int(config.max_seq_len)} "
        f"curriculum_stage={str(config.curriculum_stage or '')!r} "
        f"selection={str(config.selection)!r} "
        f"crop_policy={str(config.crop_policy)!r} "
        f"train_rows={int(len(train_examples))} "
        f"eval_rows={0 if eval_examples is None else int(len(eval_examples))} "
        f"test_rows={0 if test_examples is None else int(len(test_examples))}",
        flush=True,
    )

    train_iter = SupervisedBatchIterator(
        dataset=train_examples,
        tokenizer=tokenizer,
        batch_size=int(config.batch_size),
        max_seq_len=int(config.max_seq_len),
        crop_policy=str(config.crop_policy),
        pad_to_multiple_of=int(config.pad_to_multiple_of),
        seed=int(config.seed),
        shuffle=True,
        repeat=True,
        drop_last=True,
    )
    eval_iter = (
        None
        if eval_examples is None
        else SupervisedBatchIterator(
            dataset=eval_examples,
            tokenizer=tokenizer,
            batch_size=int(config.batch_size),
            max_seq_len=int(config.max_seq_len),
            crop_policy=str(config.crop_policy),
            pad_to_multiple_of=int(config.pad_to_multiple_of),
            seed=int(config.seed) + 10_000,
            shuffle=False,
            repeat=True,
            drop_last=False,
        )
    )
    test_iter = (
        None
        if test_examples is None
        else SupervisedBatchIterator(
            dataset=test_examples,
            tokenizer=tokenizer,
            batch_size=int(config.batch_size),
            max_seq_len=int(config.max_seq_len),
            crop_policy=str(config.crop_policy),
            pad_to_multiple_of=int(config.pad_to_multiple_of),
            seed=int(config.seed) + 20_000,
            shuffle=False,
            repeat=True,
            drop_last=False,
        )
    )

    restore_iterator_states(
        resume_state=resume_state,
        expected_stage="sft",
        iterators={
            "train_iter_state": train_iter,
            "eval_iter_state": eval_iter,
            "test_iter_state": test_iter,
        },
    )
    return SftDataState(
        train_iter=maybe_prefetch_batch_iterator(train_iter, device=device),
        eval_iter=(
            None
            if eval_iter is None
            else maybe_prefetch_batch_iterator(eval_iter, device=device)
        ),
        test_iter=(
            None
            if test_iter is None
            else maybe_prefetch_batch_iterator(test_iter, device=device)
        ),
        running_loss_sum=float(
            read_resume_state_value(
                resume_state=resume_state,
                key="running_loss_sum",
                default=0.0,
                cast=float,
            )
            or 0.0
        ),
        running_tokens=int(
            read_resume_state_value(
                resume_state=resume_state,
                key="running_tokens",
                default=0,
                cast=int,
            )
            or 0
        ),
        samples_seen=int(
            read_resume_state_value(
                resume_state=resume_state,
                key="samples_seen",
                default=0,
                cast=int,
            )
            or 0
        ),
    )
__all__ = [
    "build_sft_data_state",
]
