def test_launcher_imports() -> None:
    import ml.cli.train  # noqa: F401


def test_pretrain_imports() -> None:
    import ml.training.pretrain.shard_builder  # noqa: F401
    import ml.tasks.pretrain.pipeline  # noqa: F401
