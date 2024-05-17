"""Install script for setuptools."""

import datetime
import shutil
import subprocess
from pathlib import Path

import setuptools
from setuptools import setup
from setuptools.command import build_ext

copyright_year = datetime.date.today().strftime("%Y")


class BazelExtension(setuptools.Extension):
    """A C/C++ extension that is defined as a Bazel BUILD target."""

    def __init__(self, name: str, bazel_target: str):
        """__init__."""
        super().__init__(name=name, sources=[])

        self.bazel_target = bazel_target
        stripped_target = bazel_target.split("//")[-1]
        self.relpath, self.target_name = stripped_target.split(":")


class BazelBuildExtension(build_ext.build_ext):
    """A command that runs Bazel to build a C/C++ extension."""

    def run(self):
        """Run the build process."""
        for ext in self.extensions:
            self.bazel_build(ext)
        super().run()
        # explicitly call `bazel shutdown` for graceful exit
        self.spawn(["bazel", "shutdown"])

    def copy_extensions_to_source(self):
        """Copy generated extensions into the source tree.

        This is done in the ``bazel_build`` method, so it's not necessary to
        do again in the `build_ext` base class.
        """
        pass

    def bazel_build(self, bazel_extension: BazelExtension) -> None:
        """Run the bazel build to create the package."""
        temp_path = Path(self.build_temp)

        bazel_argv = [
            "bazel",
            "build",
            bazel_extension.bazel_target,
            f"--symlink_prefix={temp_path / 'bazel-'}",
        ]

        self.spawn(bazel_argv)

        shared_lib_suffix = ".so"
        ext_name = bazel_extension.name + shared_lib_suffix
        target_relative_path = bazel_extension.relpath + "/" + bazel_extension.name
        ext_bazel_bin_path = (
            temp_path / "bazel-bin" / bazel_extension.relpath / ext_name
        )

        ext_dest_path = self.get_ext_fullpath(target_relative_path)
        if "python/" in ext_dest_path:
            ext_dest_path = ext_dest_path.replace("python/", "")

        shutil.copyfile(ext_bazel_bin_path, Path(ext_dest_path))


class GrabBazelExtensions:
    """Class needed to grab bazel targets and convert themo to bazel extensions."""

    def __init__(self):
        """Init."""
        query = 'bazel query //beads_gym/... | grep -E "\.so"'
        output = self.run_bazel_query(query)
        self.extensions = self.process_bazel_output(output)

    def run_bazel_query(self, query: str):
        """Run the bazel query to search for .so files."""
        try:
            # Execute the Bazel command in the shell and capture the output
            result = subprocess.run(
                query,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Error executing Bazel query") from e
        except Exception as e:
            raise RuntimeError("An error occurred while running bazel query") from e

    def process_bazel_output(self, output: str):
        """Grab the bazel query output to build the BazelExtension object list."""
        try:
            lines = output.strip().split("\n")
            extensions = []
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    extension_name = parts[1].split(".")[0].split("/")[-1]
                    extension = BazelExtension(extension_name, line)
                    extensions.append(extension)
            return extensions
        except Exception as e:
            raise RuntimeError("An error occurred while processing bazel output") from e


setup(
    name="beads_gym",
    ext_modules=GrabBazelExtensions().extensions,
    cmdclass=dict(build_ext=BazelBuildExtension),
    packages=setuptools.find_packages(),
)
