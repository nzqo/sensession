"""
Helpers to facilitate remote access, mainly executing commands through remote access.
"""

from pathlib import Path
from ipaddress import IPv4Address
from dataclasses import dataclass


# fmt: off
@dataclass
class SshPasswordless:
    """
    Configuration for passwordless reemote access via ssh.

    Requires a corresponding setup in ~/.ssh/config
    """

    remote_ssh_hostname: str  # Hostname (per ssh config) for passwordless access
    pseudo_tty: bool = True   # Whether to force pseudo-tty allocation. Required to pipe SIGINT
# fmt: on


class Command:
    """
    Command wrapper class with helper generations for execution with remote devices
    """

    def __init__(self, cmd: str):
        self.cmd = cmd

    def get(self) -> str:
        """
        Get the internal original raw command
        """
        return self.cmd

    def on_remote(
        self, remote_access_config: SshPasswordless | None, pseudo_tty: bool = True
    ) -> str:
        """
        Create command to execute _on_ remote. That is, semantically equivalent:

        > login remote
        > remote: execute command

        Args:
            remote_access_config : Configuration for access to remote machine
            pseudo_tty           : Whether to force pseudo-tty allocation. Required to pipe SIGINT
        """
        if not remote_access_config:
            return self.cmd

        hostname = remote_access_config.remote_ssh_hostname
        ssh_prefix = "ssh"
        if pseudo_tty:
            ssh_prefix = f"{ssh_prefix} -tt"
        marshall_cmd = f"{ssh_prefix} {hostname} '{self.cmd}'"
        return marshall_cmd

    def script_through_remote(
        self, remote_access_config: SshPasswordless | IPv4Address | None
    ) -> str:
        """
        Create command to execute steps of a _local_ script on a remote. Semantically
        equivalent to:

        > for step in script:
        >    exec step on remote

        Args:
            remote_access_config : Configuration for access to remote machine
        """

        if not remote_access_config:
            return self.cmd

        if isinstance(remote_access_config, SshPasswordless):
            hostname = remote_access_config.remote_ssh_hostname
        elif isinstance(remote_access_config, IPv4Address):
            hostname = f"{remote_access_config}"
        else:
            raise RuntimeError(
                f"Unexpected type for remote access config: {type(remote_access_config)}"
            )

        marshall_cmd = f"ssh {hostname} 'bash -s' -- < {self.cmd}"
        return marshall_cmd


def create_marshall_copy_command(
    remote_access_config: SshPasswordless,
    from_path_remote: Path | str,
    to_path_local: Path | str,
) -> str:
    """
    Copy (recursively) file or directory on remote to local

    Args:
        remote_access_config : Config to perform remote access
        from_path_remote     : Path on the remote from where to copy
        to_path_local        : Path on the local machine to copy to
    """
    hostname = remote_access_config.remote_ssh_hostname
    return f"rsync -havzP {hostname}:{from_path_remote} {to_path_local}"
