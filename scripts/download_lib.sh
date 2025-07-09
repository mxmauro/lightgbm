#!/usr/bin/env bash

_lib_version=v4.6.0

saved_cwd=$(pwd)
cd ..
rm -r -f src1
mkdir src1
cd src1
git clone --branch $_lib_version --depth 1 https://github.com/microsoft/LightGBM .
exit_code=$?
if [ $exit_code -eq 0 ]; then
	git submodule update --init --recursive
	exit_code=$?
fi
if [ $exit_code -eq 0 ]; then
	# Remove boost from compute submodule because it gives troubles
	rm -r -f external_libs/compute/include/boost
fi
cd $saved_cwd

exit $exit_code
