import pytest

from app.schemas.tip import TipCommand
from app.services.tip import TipValidationError, check, normalize, verify


def cmd(**kw: object) -> TipCommand:
    base: dict[str, object] = {
        "action": "tip",
        "amount": 10.0,
        "is_fiat_denominated": True,
        "asset_ticker": "SOL",
        "target_user_id": "crek",
    }
    base.update(kw)
    return TipCommand.model_validate(base)


def test_normalize_canonicalises() -> None:
    out = normalize(cmd(action="  Tip ", asset_ticker="$sol", target_user_id="@Crek"))
    assert out.action == "tip"
    assert out.asset_ticker == "SOL"
    assert out.target_user_id == "Crek"


def test_valid_command_passes() -> None:
    verified = verify(cmd())
    assert verified.asset_ticker == "SOL"
    assert verified.target_user_id == "crek"
    assert verified.requires_price_conversion is True  # fiat-denominated


def test_crypto_denominated_needs_no_conversion() -> None:
    verified = verify(cmd(is_fiat_denominated=False, amount=2.5))
    assert verified.requires_price_conversion is False


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"action": "drain"}, "action"),
        ({"amount": 0.0}, "greater than zero"),
        ({"amount": -5.0}, "greater than zero"),
        ({"amount": float("inf")}, "finite"),
        ({"amount": float("nan")}, "finite"),
        ({"amount": 2_000_000.0}, "maximum"),
        ({"asset_ticker": "DOGE"}, "not a supported asset"),
        ({"target_user_id": "a"}, "target_user_id"),
        ({"target_user_id": "bad handle!"}, "target_user_id"),
    ],
)
def test_invalid_commands_are_caught(overrides: dict[str, object], needle: str) -> None:
    problems = check(normalize(cmd(**overrides)))
    assert any(needle in p for p in problems), problems


def test_verify_raises_with_problem_details() -> None:
    with pytest.raises(TipValidationError) as exc_info:
        verify(cmd(asset_ticker="DOGE", target_user_id="x"))
    details = exc_info.value.details
    assert isinstance(details, dict)
    assert len(details["problems"]) == 2  # unknown asset + bad handle
    assert details["normalized"]["asset_ticker"] == "DOGE"
