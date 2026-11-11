import json
from pathlib import Path
import pytest
from posteriorforge.campaign import campaign_plan


def test_campaign_covers_complete_paired_matrix_without_submitting(tmp_path):
    config=Path(__file__).parents[1]/'configs'/'baseline.json'
    plan=campaign_plan(config,banks='banks',references='refs',runs='runs')
    assert len(plan['references'])==8
    assert len(plan['first_gate_runs'])==24
    assert len(plan['after_gate_baseline_runs'])==48
    assert len({run['output'] for run in plan['first_gate_runs']+plan['after_gate_baseline_runs']})==72
    assert plan['submits_jobs'] is False
    value=json.loads(config.read_text());value['posterior']['clip_samples']=True
    path=tmp_path/'bad.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError):campaign_plan(path,banks='b',references='r',runs='s')
