#!/usr/bin/env python
# encoding: utf-8

from traceback import format_exception
import cProfile
import io
import pstats
import sys


def start_profiling ():
    """start profiling"""
    pr = cProfile.Profile()
    pr.enable()

    return pr


def stop_profiling (pr):
    """stop profiling and report"""
    pr.disable()

    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)

    ps.print_stats()
    print(s.getvalue())


def report_error (cause_string, logger=None, fatal=False):
    """
    TODO: errors should go to logger, and not be fatal
    """
    etype, value, tb = sys.exc_info()
    error_str = "{} {}".format(cause_string, str(format_exception(etype, value, tb, 3)))

    if logger:
        logger.info(error_str)
    else:
        print(error_str)

    if fatal:
        sys.exit(-1)
