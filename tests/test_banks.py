import json
from hashlib import sha256

import numpy as np
import pytest

from posteriorforge.banks import (
    AnalyticalBankProvider, PairedBank, _write_bank, derive_seed, load_bank,
)


def test_seeds_separate_roles_cases_ranks_repeats():
    base = derive_seed(1, "inference")
    assert base == derive_seed(1, "inference")
    assert len({base, derive_seed(1, "reference"), derive_seed(1, "inference", case=1),
                derive_seed(1, "inference", rank=1), derive_seed(1, "inference", repetition=1)}) == 5
    with pytest.raises(ValueError):
        derive_seed(-1, "inference")


def test_paired_bank_authentication_and_nested_prefixes(tmp_path):
    bank = AnalyticalBankProvider().get_bank(seed=20261111, count=1024)
    path = tmp_path / "bank.npz"
    _write_bank(path, bank)
    loaded = load_bank(path)
    for n in (256, 1024):
        prefix = loaded.prefix(n)
        np.testing.assert_array_equal(prefix.theta, bank.theta[:n])
        np.testing.assert_array_equal(prefix.obs, bank.obs[:n])
    with pytest.raises(ValueError):
        loaded.theta.flags.writeable = True
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        load_bank(path)


def test_invalid_complete_rows_and_unsupported_noise_rejected():
    bank = AnalyticalBankProvider().get_bank(seed=4, count=2)
    with pytest.raises(ValueError, match="paired"):
        PairedBank(bank.theta[:, :20], bank.obs, {})
    invalid = bank.theta.copy()
    invalid[0, 6] = .501
    with pytest.raises(ValueError, match="prior support"):
        PairedBank(invalid, bank.obs, {})
    invalid[0, 6] = np.nan
    with pytest.raises(ValueError, match="finite"):
        PairedBank(invalid, bank.obs, {})
