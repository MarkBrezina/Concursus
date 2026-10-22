"""Finance: placeholder — add your limit-order-book tests here as the module grows."""


def test_finance_imports():
    from stochastics import finance
    from stochastics.finance import lob

    assert finance.__name__ == 'stochastics.finance' and lob.__name__ == 'stochastics.finance.lob'
