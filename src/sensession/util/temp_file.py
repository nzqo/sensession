"""
Management of temporary files
"""

from pathlib import Path

from loguru import logger

from sensession.config import APP_CONFIG


class TempFile:
    """
    A class helper for managing temporary files
    """

    def __init__(self, file_name: str, file_dir: Path = APP_CONFIG.cache_dir / "tmp"):
        file_dir.mkdir(exist_ok=True, parents=True)
        self._file: Path = file_dir / f"{file_name}"

        if self._file.is_file() and not APP_CONFIG.overwrite_tmp_files:
            raise FileExistsError(
                f"{self._file} already exists; Choose a different name or directory!"
            )

        if self._file.is_file():
            self._file.unlink()

        logger.debug(f"Created TempFile object for file: {self._file}")

    def close(self):
        """
        Close the temporary file
        """
        logger.trace(f"Closing temp file {self._file}")

        if APP_CONFIG.keep_tmp_files:
            return

        if self._file.is_file():
            logger.debug(f"Removing temporary file: {self._file}")
            self._file.unlink()

    def __enter__(self):
        """
        Context entering. Trivial since the file is opened in the constructor.
        """
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Context exit. Ensure that subprocesses are properly terminated.
        """
        self.close()

        if exc_type is not None:
            logger.error("Encountered error in TempFile context ...")
            return False

        return True

    def empty(self) -> bool:
        """
        Check whether file is empty
        """
        return (not self._file.is_file()) or self._file.stat().st_size == 0

    @property
    def path(self) -> Path:
        """
        Get the filepath to the tempfile to use
        """
        return self._file
