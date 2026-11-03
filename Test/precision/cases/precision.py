"""Rules from opbase 79f865595952779aa3fd6c3069e1638cf501f3f0.

Integer/index outputs use exact equality as their business rule. The upstream
mixed tolerance standard applies to floating point computation only. Double
has no upstream tolerance entry, so it uses exact comparison here unless an
operator-specific rule is agreed. Shape mismatches always fail; floating
golden values may have a higher dtype than the actual output.
"""
import torch


def check_precision(actual, golden):
    actual = actual.detach().cpu()
    golden = golden.detach().cpu()
    if actual.shape != golden.shape:
        return dict(passed=False, reason='shape mismatch')
    if not actual.is_floating_point() or actual.dtype == torch.float64:
        # Never cast indices to float: int64 IDs above 2**53 must stay exact.
        passed = actual.dtype == golden.dtype and torch.equal(actual, golden)
        return dict(passed=passed, rule='exact', elements=actual.numel())
    thresholds = {
        torch.float16: (2**-10, .1),
        torch.bfloat16: (2**-7, .1),
        torch.float32: (2**-13, .01),
    }
    if actual.dtype not in thresholds:
        raise ValueError(f'No precision rule for {actual.dtype}')
    tolerance, floor = thresholds[actual.dtype]
    a, g = actual.double(), golden.double()
    narrowed = golden.to(actual.dtype).double()
    finite = torch.isfinite(a) & torch.isfinite(g)
    error = (a - g).abs()
    matched = finite & (error <= tolerance + tolerance * g.abs())
    # Finite golden that overflows on narrowing is allowed to match same-sign
    # infinity; finite actual is always compared to the high precision golden.
    matched |= torch.isinf(a) & torch.isinf(narrowed) & (a.sign() == narrowed.sign())
    matched |= torch.isnan(a) & torch.isnan(g)
    ratio = matched.double().mean().item() if matched.numel() else 1.
    limit, max_error = floor, 0.
    if finite.any():
        errors = error[finite]
        index = errors.argmax()
        max_error = errors[index].item()
        anchor = golden[finite][index].to(actual.dtype).abs()
        ulp = torch.nextafter(anchor, torch.full_like(anchor, float('inf'))) - anchor
        if not torch.isfinite(ulp):
            ulp = anchor - torch.nextafter(anchor, torch.zeros_like(anchor))
        limit = max(floor, 32 * ulp.double().item())
    return dict(passed=ratio >= .99 and max_error <= limit, rule='mixed_tolerance',
                matched_ratio=ratio, max_abs_error=max_error, max_abs_error_limit=limit,
                rtol=tolerance, atol=tolerance, elements=actual.numel())
