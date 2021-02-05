#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:24:54 2020

@author: mdeprez
"""
from subprocess import check_output


def execute_process(command_shell):
	stdout = check_output(command_shell, shell=True).strip()
	if not isinstance(stdout, str):
		stdout = stdout.decode()
	return stdout


__all__ = [
	'execute_process'
]
