import getpass
import multiprocessing
import os
import sys


sys.path.append("/work/asr4/raissi/dev/cachemanager/")

WORK_DIR = "work"
IMPORT_PATHS = ["config", "recipe", "recipe/"]


def file_caching(path):
    return "`cf %s`" % path


def engine():
    from sisyphus.localengine import LocalEngine
    from sisyphus.engine import EngineSelector
    from sisyphus.son_of_grid_engine import SonOfGridEngine

    default_rqmt = {
        "cpu": 1,
        "mem": 1,
        "gpu": 0,
        "time": 1,
        "qsub_args": "-l hostname=!(cluster-cn-224|cluster-cn-243)",
    }

    return EngineSelector(
        engines={
            "short": LocalEngine(cpus=multiprocessing.cpu_count()),
            "long": SonOfGridEngine(default_rqmt=default_rqmt),
        },
        default_engine="long",
    )


MAIL_ADDRESS = getpass.getuser()

# SIS_COMMAND = ['/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1/bin/python3.8', sys.argv[0]]
SIS_COMMAND = [
    "/work/asr3/rossenbach/schuemann/new_simple_tts_env/bin/python",
    sys.argv[0],
]

WAIT_PERIOD_CACHE = 8
WAIT_PERIOD_JOB_FS_SYNC = 8

JOB_CLEANUP_KEEP_WORK = True
JOB_FINAL_LOG = "finished.tar.gz"

SHOW_JOB_TARGETS = False
PRINT_ERROR_TASKS = 0
PRINT_ERROR_LINES = 10

RASR_ROOT = "/u/raissi/dev/rasr_tf14py38_fh/"
# RASR_ROOT = '/work/tools/asr/rasr/20191102_generic/'
# RASR_ROOT = '/u/raissi/dev/thesis-rasr-dense/'
# RASR_ROOT = '/u/raissi/dev/rasr-dense/'

RASR_ARCH = "linux-x86_64-standard"

SCTK_PATH = "/u/beck/programs/sctk-2.4.0/bin/"

# RETURNN_PYTHON_HOME='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1'
# RETURNN_PYTHON_EXE='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python'
# RETURNN_PYTHON_HOME='/work/tools/asr/python/3.8.0_tf_2.3.4-generic+cuda10.1+mkl'
# RETURNN_PYTHON_EXE='/work/tools/asr/python/3.8.0_tf_2.3.4-generic+cuda10.1+mkl/bin/python3'
# RETURNN_PYTHON_HOME='/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1'
RETURNN_PYTHON_EXE = (
    "/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1/bin/python3.8"
)

RASR_PYTHON_HOME = "/work/tools/asr/python/3.8.0"
RASR_PYTHON_EXE = "/work/tools/asr/python/3.8.0/bin/python3.8"

SHOW_JOB_TARGETS = False


DEFAULT_ENVIRONMENT_SET = {
    "PATH": ":".join(
        [
            "/rbi/sge/bin",
            "/rbi/sge/bin/lx-amd64",
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
            "/usr/games",
            "/usr/local/games",
            "/snap/bin",
        ]
    ),
    "LD_LIBRARY_PATH": ":".join(
        [
            "/work/tools/asr/python/3.8.0/lib/python3.8/site-packages/numpy/.libs",
            "/usr/local/cudnn-10.1-v7.6/lib64",
            "/usr/local/cuda-10.1/lib64",
            "/usr/local/cuda-10.1/extras/CUPTI/lib64/",
            "/usr/local/cuda-9.1/extras/CUPTI/lib64/",
            "/usr/local/cuda-9.1/lib64",
            "/usr/local/acml-4.4.0/cblas_mp/lib",
            "/usr/local/acml-4.4.0/gfortran64_mp/lib/",
            "/usr/local/acml-4.4.0/gfortran64/lib",
            "/usr/lib/nvidia-418",
            "/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/lib/python3.8/site-packages/tensorflow",
        ]
    ),
    "CUDA_PATH": ":".join(
        ["/usr/local/cuda-9.1", "/usr/local/cuda-10.1", "/usr/local/cudnn-10.1-v7.6/"]
    ),
    "TMPDIR": "/var/tmp",
}


def check_engine_limits(current_rqmt, task):
    """Check if requested requirements break and hardware limits and reduce them.
    By default ignored, a possible check for limits could look like this::

        current_rqmt['time'] = min(168, current_rqmt.get('time', 2))
        if current_rqmt['time'] > 24:
            current_rqmt['mem'] = min(63, current_rqmt['mem'])
        else:
            current_rqmt['mem'] = min(127, current_rqmt['mem'])
        return current_rqmt

    :param dict[str] current_rqmt: requirements currently requested
    :param sisyphus.task.Task task: task that is handled
    :return: requirements updated to engine limits
    :rtype: dict[str]
    """
    current_rqmt["time"] = min(168, current_rqmt.get("time", 1))
    return current_rqmt
