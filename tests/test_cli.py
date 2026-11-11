import json
import subprocess
import sys
from posteriorforge.cli import main


def test_help_is_import_safe():
    result = subprocess.run([sys.executable, '-m', 'posteriorforge', '--help'], capture_output=True, text=True, check=True)
    assert 'prepare-banks' in result.stdout
    assert 'qualify-reference' in result.stdout


def test_doctor_does_not_initialize_model(capsys):
    main(['doctor'])
    result = json.loads(capsys.readouterr().out)
    assert result['schema'] == 'posteriorforge.runtime.v1'
    assert 'cuda' not in result
