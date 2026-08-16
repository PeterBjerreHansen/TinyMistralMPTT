from hashlib import sha256
from pathlib import Path


def test_vanilla_source_manifest_matches_checked_in_copy():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "docs" / "VANILLA_SOURCE.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        assert path.is_file(), relative
        assert sha256(path.read_bytes()).hexdigest() == expected, relative
