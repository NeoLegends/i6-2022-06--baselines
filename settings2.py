import getpass
import sys



sys.path.append('/work/asr4/raissi/dev/cachemanager/')


WORK_DIR = 'work'
IMPORT_PATHS = ['config', 'recipe', 'recipe/']
def file_caching(path):
  return '`cf %s`' % path

def engine():
  from sisyphus.localengine import LocalEngine
  from sisyphus.engine import EngineSelector
  from sisyphus.son_of_grid_engine import SonOfGridEngine

  default_rqmt = {
    'cpu'      : 1,
    'mem'      : 1,
    'gpu'      : 0,
    'time'     : 1,
    'qsub_args': '',
    }

  return EngineSelector(engines={'short': LocalEngine(cpus=3),
                                 'long' : SonOfGridEngine(default_rqmt=default_rqmt)},
                        default_engine='long')

MAIL_ADDRESS = getpass.getuser()

SIS_COMMAND = ['/u/raissi/bin/opt/sisyphus/bin/python3.8', sys.argv[0]]



WAIT_PERIOD_CACHE = 8
WAIT_PERIOD_JOB_FS_SYNC = 8

JOB_CLEANUP_KEEP_WORK = True
JOB_FINAL_LOG = 'finished.tar.gz'

SHOW_JOB_TARGETS = False
PRINT_ERROR_TASKS = 0
PRINT_ERROR_LINES = 10

RASR_ROOT = '/work/tools/asr/rasr/20191102_generic/'
#RASR_ROOT = '/u/raissi/dev/thesis-rasr-dense/'
#RASR_ROOT = '/u/raissi/dev/rasr-dense/'

RASR_ARCH = 'linux-x86_64-standard'

SCTK_PATH = '/u/beck/programs/sctk-2.4.0/bin/'

G2P_PATH = '/u/beck/dev/g2p/release/lib/python/g2p.py'
G2P_PYTHON = 'python2.7'

#RETURNN_PYTHON_HOME='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1'
#RETURNN_PYTHON_EXE='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python'
#RETURNN_PYTHON_HOME='/work/tools/asr/python/3.8.0_tf_2.3.4-generic+cuda10.1+mkl'
#RETURNN_PYTHON_EXE='/work/tools/asr/python/3.8.0_tf_2.3.4-generic+cuda10.1+mkl/bin/python3'
RETURNN_PYTHON_HOME='/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1'
RETURNN_PYTHON_EXE='/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1/bin/python3.8'
#RETURNN_PYTHON_HOME = '/u/beck/programs/python/2.7.10/'
#RETURNN_PYTHON_EXE  = '/u/beck/programs/python/2.7.10/bin/python2.7'
#RETURNN_ROOT        = '/u/raissi/setups/swb/crnn/'
RETURNN_ROOT        = '/u/raissi/dev/returnn/'

RASR_PYTHON_HOME = '/u/beck/programs/python/2.7.10/'
RASR_PYTHON_EXE  = '/u/beck/programs/python/2.7.10/bin/python2.7'

SHOW_JOB_TARGETS = False

"""
DEFAULT_ENVIRONMENT_SET = { 'PATH'            : ':'.join(['/rbi/sge/bin', '/rbi/sge/bin/lx-amd64',
                                                          '/usr/local/sbin', '/usr/local/bin',
                                                          '/usr/sbin', '/usr/bin',
                                                          '/sbin', '/bin',
                                                          '/usr/games', '/usr/local/games',
                                                          '/snap/bin']),
                            'LD_LIBRARY_PATH' : ':'.join(['/usr/local/cuda-9.1/extras/CUPTI/lib64/',
                                                          '/usr/local/cuda-9.1/lib64',
                                                          '/usr/local/cudnn-9.1-v7.1/lib64',
                                                          '/usr/local/acml-4.4.0/cblas_mp/lib',
                                                          '/usr/local/acml-4.4.0/gfortran64_mp/lib/',
                                                          '/usr/local/acml-4.4.0/gfortran64/lib']),

                            'CUDA_PATH'       : ':'.join(['/usr/local/cuda-9.1']),
                            'TMPDIR'          : '/var/tmp'}
"""

DEFAULT_ENVIRONMENT_SET = {'MKL_NUM_THREADS': 1,
                           'OMP_NUM_THREADS': 1,
                           'PATH': ':'.join(['/usr/local/cuda-10.1/bin:'
                                             '/rbi/sge/bin', '/rbi/sge/bin/lx-amd64',
                                             '/usr/local/sbin', '/usr/local/bin',
                                             '/usr/sbin', '/usr/bin',
                                             '/sbin', '/bin',
                                             '/usr/games', '/usr/local/games',
                                             '/snap/bin']),
                           'LD_LIBRARY_PATH': ':'.join(['/usr/local/acml-4.4.0/cblas_mp/lib',
                                                        '/usr/local/acml-4.4.0/gfortran64_mp/lib/',
                                                        '/usr/local/acml-4.4.0/gfortran64/lib',
                                                        '/usr/local/cuda-10.1/extras/CUPTI/lib64/',
                                                        '/usr/local/cuda-10.1/lib64',
                                                        '/usr/local/cudnn-10.1-v7.6/lib64']),
                           'CUDA_PATH': '/usr/local/cuda-10.1',
                           'TMPDIR': '/var/tmp'}

