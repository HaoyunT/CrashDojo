"""Keep the native run_job audit while replacing only its eval entry in a trial."""

from pathlib import Path
import run_job


ENTRY = Path('/path/to/workspace/Friday/safesim_planner_20261012_01/launch/baselines_native_v6_remaining225_20261101/ddpm_compact_entry.py')
original_command = run_job.command


def compact_command(job, output, steps):
    assert job['method'] == 'ccdiff_ddpm'
    command = original_command(job, output, steps)
    command[1] = str(ENTRY)
    return command


run_job.command = compact_command
run_job.main()
