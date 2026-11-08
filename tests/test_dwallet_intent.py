import pytest

from app.schemas.dwallet_intent import IntentRejected, ParsedIntent, settle


def intent(**kw: object) -> ParsedIntent:
    base: dict[str, object] = {
        "action": "tip",
        "amount": 5.0,
        "is_fiat_denominated": False,
        "asset": "SOL",
        "recipient": "crek",
    }
    base.update(kw)
    return ParsedIntent.model_validate(base)


def test_named_recipient_settles() -> None:
    out = settle(intent())
    assert out.recipient == "crek"
    assert out.requires_price_conversion is False


def test_fiat_flags_conversion() -> None:
    out = settle(intent(is_fiat_denominated=True, amount=10.0, asset="USDC"))
    assert out.requires_price_conversion is True


def test_split_settles() -> None:
    out = settle(intent(recipient=None, split_count=5, asset="USDC", amount=10.0))
    assert out.split_count == 5
    assert out.recipient is None


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"amount": 0.0}, "greater than zero"),
        ({"amount": -1.0}, "greater than zero"),
        ({"amount": float("inf")}, "finite"),
        ({"amount": 2_000_000.0}, "maximum"),
        ({"recipient": None, "split_count": None}, "no recipient"),
        ({"recipient": "crek", "split_count": 3}, "ambiguous"),
        ({"recipient": "!"}, "recipient must be"),
        ({"recipient": None, "split_count": 999}, "split_count must be"),
    ],
)
def test_rejections(overrides: dict[str, object], needle: str) -> None:
    with pytest.raises(IntentRejected) as exc:
        settle(intent(**overrides))
    assert any(needle in p for p in exc.value.problems), exc.value.problems


def test_injection_string_cannot_become_valid() -> None:
    # The model can only emit enum assets/actions; a junk ticker never type-checks.
    with pytest.raises(Exception):  # noqa: B017 - ValidationError on the Literal
        ParsedIntent.model_validate(
            {
                "action": "tip",
                "amount": 100.0,
                "is_fiat_denominated": False,
                "asset": "IGNORE_RULES_BTC",
                "recipient": "me",
            }
        )
