#!/usr/bin/env python3
"""Frozen runtime profiles for the carbon PBE reference gate."""

import argparse
from types import MappingProxyType


_PROFILE_FIELDS = (
    "name",
    "partition",
    "nodes",
    "ntasks",
    "cpus_per_task",
    "memory_mb",
    "time_limit",
    "over_subscribe",
)

_RESOURCE_PROFILES = MappingProxyType(
    {
        "df_dcu": MappingProxyType(
            {
                "name": "df_dcu",
                "partition": "normal",
                "nodes": 1,
                "ntasks": 1,
                "cpus_per_task": 30,
                "memory_mb": 110610,
                "time_limit": "1-00:00:00",
                "over_subscribe": "NO",
            }
        ),
        "server66": MappingProxyType(
            {
                "name": "server66",
                "partition": "640",
                "nodes": 1,
                "ntasks": 1,
                "cpus_per_task": 48,
                "memory_mb": 180000,
                "time_limit": "1-00:00:00",
                "over_subscribe": "OK",
            }
        ),
    }
)


def get_resource_profile(name):
    """Return an independent copy of an explicitly named runtime profile."""
    if not isinstance(name, str) or name not in _RESOURCE_PROFILES:
        raise ValueError("unknown C PBE gate profile: {!r}".format(name))
    return dict(_RESOURCE_PROFILES[name])


def _main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    shell_parser = subparsers.add_parser("shell")
    shell_parser.add_argument("name")
    arguments = parser.parse_args()

    if arguments.command != "shell":
        parser.error("a command is required")

    profile = get_resource_profile(arguments.name)
    print("|".join(str(profile[field]) for field in _PROFILE_FIELDS))


if __name__ == "__main__":
    _main()
