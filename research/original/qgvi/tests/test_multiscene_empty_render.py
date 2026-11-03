from types import SimpleNamespace

import pytest
import torch

from qgvi.multiscene_renderer import _render_allow_empty


@pytest.mark.parametrize('method', ['generate_neural_gaussians', 'generate_explicit_gaussians'])
def test_empty_generation_returns_exact_background_without_rasterizing(method):
    camera = SimpleNamespace(image_height=7, image_width=11)
    background = torch.tensor([0.2, 0.4, 0.8])
    original_generator = lambda *args: (torch.empty((0, 3)), 'other outputs')
    model = SimpleNamespace(**{method: original_generator}, arbitrary_state='preserved')

    def renderer(camera, guarded, pipeline, bg):
        assert guarded.arbitrary_state == 'preserved'
        getattr(guarded, method)(camera)
        pytest.fail('Empty primitives must never reach the rasterizer')

    result = _render_allow_empty(renderer, camera, model, None, background)
    assert result['empty_gaussian_set'] is True
    assert result['render'].shape == (3, 7, 11)
    assert torch.equal(result['render'], background[:, None, None].expand(3, 7, 11))
    assert result['render_depth'].shape == result['render_alphas'].shape == (1, 7, 11)
    assert torch.count_nonzero(result['render_depth']) == 0
    assert torch.count_nonzero(result['render_alphas']) == 0
    assert getattr(model, method) is original_generator


def test_nonempty_generation_and_renderer_result_are_unchanged():
    means = torch.ones((2, 3))
    generated = (means, 'other outputs')
    result = {'render': 'original rasterizer result'}
    model = SimpleNamespace(generate_neural_gaussians=lambda: generated)

    def renderer(camera, guarded, pipeline, bg):
        assert guarded.generate_neural_gaussians() is generated
        return result

    assert _render_allow_empty(renderer, None, model, None, None) is result


def test_other_renderer_failures_are_not_converted_to_empty_background():
    def renderer(*args):
        raise ValueError('invalid rasterizer output')

    with pytest.raises(ValueError, match='invalid rasterizer output'):
        _render_allow_empty(renderer, None, object(), None, None)
