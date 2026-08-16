from tiny_mistral_mptt.data.packed_dataset import StatefulBlockSampler


def test_sampler_resume_reproduces_exact_next_batches():
    original = StatefulBlockSampler(17, seed=99)
    original.next_indices(9)
    state = original.state_dict()
    expected = [original.next_indices(5), original.next_indices(20)]

    resumed = StatefulBlockSampler(17, seed=123456)
    resumed.load_state_dict(state)
    actual = [resumed.next_indices(5), resumed.next_indices(20)]
    assert actual == expected
