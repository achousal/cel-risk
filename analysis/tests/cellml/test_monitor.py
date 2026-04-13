"""Monitor tests — mock bjobs stdout."""

from __future__ import annotations

from unittest.mock import MagicMock

from ced_ml.cellml.monitor import _parse_bjobs_wide, get_status

SAMPLE_BJOBS = """\
JOBID      USER     STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME             SUBMIT_TIME
100001     achous   RUN   premium    login1      node5       my_exp[1]            Apr 13 10:00
100002     achous   RUN   premium    login1      node6       my_exp[2]            Apr 13 10:00
100003     achous   PEND  premium    login1      -           my_exp[3]            Apr 13 10:00
100004     achous   DONE  premium    login1      node7       my_exp[4]            Apr 13 10:00
100005     achous   EXIT  premium    login1      node8       my_exp[5]            Apr 13 10:00
100006     achous   RUN   premium    login1      node9       other_exp[1]         Apr 13 10:00
"""


def test_parse_counts_states():
    counts = _parse_bjobs_wide(SAMPLE_BJOBS, "my_exp")
    assert counts["PEND"] == 1
    assert counts["RUN"] == 2
    assert counts["DONE"] == 1
    assert counts["EXIT"] == 1
    # other_exp is filtered out
    assert sum(counts.values()) == 5


def test_parse_empty_stdout():
    counts = _parse_bjobs_wide("", "exp")
    assert all(v == 0 for v in counts.values())


def test_get_status_mocks_bjobs():
    fake = MagicMock()
    fake.return_value = MagicMock(stdout=SAMPLE_BJOBS, stderr="", returncode=0)
    result = get_status("my_exp", runner=fake, env={"USER": "achous"})
    assert result.error is None
    assert result.counts["RUN"] == 2
    assert result.counts["PEND"] == 1
    fake.assert_called_once()
    # Verify -w and -u were passed
    args = fake.call_args[0][0]
    assert "bjobs" in args[0]
    assert "-w" in args
    assert "-u" in args


def test_get_status_filenotfound_graceful():
    def raiser(*args, **kwargs):
        raise FileNotFoundError("bjobs missing")

    result = get_status("exp", runner=raiser, env={"USER": "x"})
    assert result.error is not None
    assert all(v == 0 for v in result.counts.values())
