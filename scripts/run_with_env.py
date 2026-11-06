#!/usr/bin/env python3
"""Run an analysis script/CLI using this installed checkout's environment.

Usage: .venv/bin/python scripts/run_with_env.py [-H] [-n 4] SCRIPT [ARGS...]
       .venv/bin/python scripts/run_with_env.py -- -m chromorph doctor
No source-directory PYTHONPATH is required: install ChroMorph with pip first.
"""
from pathlib import Path
import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-H', '--high-resolution', action='store_true')
    parser.add_argument('-n', '--mpi-ranks', type=int, default=1)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:parser.error('Provide a Python script or -m MODULE')
    python = root/'.venv/bin/python'
    if not python.exists():parser.error('Create .venv and pip install this package first')
    env = dict(os.environ)
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('PYSYN_CDBS', 'https://ssb.stsci.edu/trds/')
    if (root/'fsps-data').is_dir():env.setdefault('SPS_HOME', str(root/'fsps-data'))
    mpi = root/'vendor/mpich'
    if mpi.is_dir():
        env['PATH'] = str(mpi/'bin')+os.pathsep+env.get('PATH', '')
        env['LD_LIBRARY_PATH'] = str(mpi/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH', '')
    if args.high_resolution:
        backend = root/'vendor/fsps_c3k_hr/python'
        if not backend.is_dir():parser.error('Install an isolated C3K_HR backend in vendor/fsps_c3k_hr/python')
        env['PYTHONPATH'] = str(backend)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    argv = [str(python), *command]
    if args.mpi_ranks != 1:
        if args.mpi_ranks < 1:parser.error('MPI ranks must be positive')
        argv = [str(mpi/'bin/mpiexec'), '-n', str(args.mpi_ranks), *argv]
    raise SystemExit(subprocess.call(argv, env=env))


if __name__ == '__main__':main()
