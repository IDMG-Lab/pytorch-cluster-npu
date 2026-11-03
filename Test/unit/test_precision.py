import torch
from precision import check_precision


def test_integer_ids_are_not_rounded():
    assert not check_precision(torch.tensor([2**53+1]), torch.tensor([2**53]))['passed']


def test_hard_limit_intercepts_outlier_with_99_percent_matches():
    a = torch.zeros(100)
    a[0] = .02
    result = check_precision(a, torch.zeros(100, dtype=torch.float64))
    assert result['matched_ratio'] == .99
    assert not result['passed']


def test_dynamic_ulp_limit_uses_largest_error_anchor():
    golden = torch.tensor([1e8, 0.], dtype=torch.float64)
    a = torch.tensor([1e8+128, 0.], dtype=torch.float32)
    result = check_precision(a, golden)
    assert result['passed'] and result['max_abs_error_limit'] == 256


def test_overflow_and_special_values():
    a = torch.tensor([float('inf'), -float('inf'), float('nan')], dtype=torch.float16)
    golden = torch.tensor([1e9, -1e9, float('nan')], dtype=torch.float64)
    assert check_precision(a, golden)['passed']
    assert not check_precision(-a[:1], golden[:1])['passed']


def test_empty_and_shape_mismatch():
    assert check_precision(torch.empty(0), torch.empty(0))['passed']
    assert not check_precision(torch.empty(0), torch.empty(1))['passed']
