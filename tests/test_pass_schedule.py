from tiny_mistral_mptt.training.pass_schedule import PassScheduler


def test_pass_scheduler_state_restores_exact_future_draws():
    stages = [
        {"until_tokens": 100, "probabilities": {2: 1.0}},
        {"probabilities": {1: 0.5, 2: 0.4, 3: 0.1}},
    ]
    scheduler = PassScheduler(stages, seed=19)
    assert [scheduler.sample(tokens) for tokens in (0, 50, 99)] == [2, 2, 2]
    state = scheduler.state_dict()
    expected = [scheduler.sample(100) for _ in range(20)]

    restored = PassScheduler(stages, seed=999)
    restored.load_state_dict(state)
    assert [restored.sample(100) for _ in range(20)] == expected


def test_pass_scheduler_normalizes_probabilities():
    scheduler = PassScheduler([{"probabilities": {1: 1, 2: 3}}], seed=1)
    assert scheduler.stages[0]["probabilities"] == {1: 0.25, 2: 0.75}
