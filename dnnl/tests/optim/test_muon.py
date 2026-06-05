import inspect

import pytest
import torch

import dnnl.optim as dopt
import dnnl.optim.muon as muon


def test_muon_module_has_docstrings():
    for name in muon.__all__:
        member = getattr(muon, name)
        assert inspect.getdoc(member), name

        for method_name, method in inspect.getmembers(member, inspect.isfunction):
            if method.__qualname__.startswith(f'{member.__name__}.'):
                assert inspect.getdoc(method), f'{name}.{method_name}'

    assert inspect.getdoc(muon.newton_schulz_5)


def test_muon_public_export():
    assert dopt.Muon is muon.Muon


def test_newton_schulz_5_preserves_tall_matrix_shape():
    update = torch.tensor([[3.0, 0.0], [4.0, 0.0], [0.0, 0.0]])

    actual = muon.newton_schulz_5(
        update,
        ns_steps=1,
        eps=0.0,
        ns_coefficients=(1.0, 0.0, 0.0),
    )

    assert actual.shape == update.shape
    assert torch.allclose(actual, update / update.norm())


def test_muon_rejects_non_matrix_parameters():
    param = torch.tensor([1.0, 2.0], requires_grad=True)

    with pytest.raises(NotImplementedError, match='2D parameters'):
        dopt.Muon([param])


def test_muon_accumulates_momentum_and_updates_parameters():
    param = torch.tensor([[1.0, -2.0], [0.5, 1.0]], requires_grad=True)
    optimizer = dopt.Muon(
        [param],
        lr=0.1,
        momentum=0.5,
        ns_steps=1,
        eps=0.0,
        ns_coefficients=(1.0, 0.0, 0.0),
    )

    param.grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    optimizer.step()

    assert torch.allclose(optimizer.momentum_buffers[0], param.grad)
    assert torch.allclose(param, torch.tensor([[0.94, -2.08], [0.5, 1.0]]))

    param.grad = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
    optimizer.step()

    expected_buffer = torch.tensor([[1.5, 2.0], [0.0, 5.0]])
    expected_update = expected_buffer / expected_buffer.norm()
    assert torch.allclose(optimizer.momentum_buffers[0], expected_buffer)
    assert torch.allclose(
        param,
        torch.tensor([[0.94, -2.08], [0.5, 1.0]]) - 0.1 * expected_update,
    )


def test_muon_skips_parameters_without_gradients():
    trained = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    skipped = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    trained.grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    optimizer = dopt.Muon(
        [trained, skipped],
        lr=0.1,
        ns_steps=1,
        eps=0.0,
        ns_coefficients=(1.0, 0.0, 0.0),
    )

    optimizer.step()

    assert torch.allclose(trained, torch.tensor([[0.94, -0.08], [0.0, 1.0]]))
    assert torch.allclose(skipped, torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    assert torch.equal(optimizer.momentum_buffers[1], torch.zeros_like(skipped))
