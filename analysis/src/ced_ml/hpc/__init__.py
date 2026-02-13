"""HPC job submission utilities.

Re-exports the public API so callers can do::

    from ced_ml.hpc import build_job_script, submit_job, get_scheduler
"""

from ced_ml.hpc.base import SchedulerBackend, get_scheduler  # noqa: F401
from ced_ml.hpc.common import (  # noqa: F401
    EnvironmentInfo,
    build_job_script,
    detect_environment,
    load_hpc_config,
    submit_hpc_pipeline,
    submit_job,
    validate_identifier,
)
