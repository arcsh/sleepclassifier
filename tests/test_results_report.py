"""Results report generation tests."""

from __future__ import annotations

from sleepstage.evaluation.results_report import format_status_line, write_results_md


def test_results_status_with_fixes(tmp_path) -> None:
    exp = tmp_path / "experiments"
    exp.mkdir()
    (tmp_path / "logs").mkdir()
    line = format_status_line(
        "SUCCESS",
        [
            "download escalated to rsync after 2 HTTP stalls",
            "3 records skipped, see skipped_records.log",
        ],
    )
    assert "rsync" in line
    write_results_md(
        tmp_path / "RESULTS.md",
        exp,
        auto_fixes=["download escalated to rsync after 2 HTTP stalls"],
    )
    assert "STATUS: FAILED (download escalated to rsync" in (tmp_path / "RESULTS.md").read_text()
