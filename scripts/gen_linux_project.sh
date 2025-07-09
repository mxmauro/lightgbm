#!/usr/bin/env bash

# Parse command line arguments
boost_include_dir=""
boost_libs_dir=""
nvidia_cuda_base_dir=""
make_static_libs="OFF"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--boost-include-dir)
			if [[ -z "$2" ]]; then
				echo "Error: --boost-include-dir requires a directory path."
				exit 1
			fi
			boost_include_dir="$2"
			shift 2
			;;
		--boost-libs-dir)
			if [[ -z "$2" ]]; then
				echo "Error: --boost-libs-dir requires a directory path."
				exit 1
			fi
			boost_libs_dir="$2"
			shift 2
			;;
		--nvidia-cuda-dir)
			if [[ -z "$2" ]]; then
				echo "Error: --nvidia-cuda-dir requires a directory path."
				exit 1
			fi
			nvidia_cuda_base_dir="$2"
			shift 2
			;;
		--make-static)
			make_static_libs="ON"
			shift 1
			;;
		*)
			echo "Error: Invalid parameter '$1'"
			exit 1
			;;
	esac
done

# Check options
if [[ "$nvidia_cuda_base_dir" == "auto" ]]; then
	_dir=/usr/local/cuda
	if [ -d "$_dir" ]; then
		echo "Info: Detected CUDA in directory '$_dir'"
		nvidia_cuda_base_dir=$_dir
	else
		nvidia_cuda_base_dir=""
	fi
fi

if [[ -z "$nvidia_cuda_base_dir" ]]; then
	cmake_cuda_options="-DUSE_GPU=OFF"
else
	# CUDA requires boost
	if [[ -z "$boost_include_dir" ]]; then
		_dir=/usr/include
		if [ -d "$_dir/boost" ]; then
			echo "Info: Detected BOOST in directory '$_dir'"
			boost_include_dir=$_dir
		else
			echo "Error: Missing --boost-include-dir argument."
			exit 1
		fi
	fi
	if [[ -z "$boost_libs_dir" ]]; then
		_dir=/usr/lib/x86_64-linux-gnu
		if [ -f "$_dir/libboost_system.a" ]; then
			echo "Info: Detected BOOST in directory '$_dir'"
			boost_libs_dir=$_dir
		else
			echo "Error: Missing --boost-libs-dir argument."
			exit 1
		fi
	fi
	

	echo "Info: Adding support for CUDA."

	cmake_cuda_options="-DUSE_GPU=ON"
	cmake_cuda_options="$cmake_cuda_options -DBoost_NO_SYSTEM_PATHS=ON"
	cmake_cuda_options="$cmake_cuda_options -DBoost_USE_STATIC_LIBS=$make_static_libs"
	cmake_cuda_options="$cmake_cuda_options -DBoost_USE_STATIC_RUNTIME=$make_static_libs"
	cmake_cuda_options="$cmake_cuda_options -DBoost_INCLUDE_DIR=$boost_include_dir"
	cmake_cuda_options="$cmake_cuda_options -DBOOST_LIBRARYDIR=$boost_libs_dir"
	cmake_cuda_options="$cmake_cuda_options -DOpenCL_LIBRARY=$nvidia_cuda_base_dir/lib64/libOpenCL.so"
	cmake_cuda_options="$cmake_cuda_options -DOpenCL_INCLUDE_DIR=$nvidia_cuda_base_dir/include/"
fi

# Build projects
saved_cwd=$(pwd)
cd ../src
rm -r -f ../build/linux
cmake -B ../build/linux/debug/project -S . -G "Unix Makefiles" \
	-DUSE_DEBUG=ON \
	-DCMAKE_BUILD_TYPE=Debug \
	-DBUILD_STATIC_LIB=$make_static_libs \
	$cmake_cuda_options \
	-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=../bin \
	-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../bin \
	-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=../lib
exit_code=$?
if [ $exit_code -eq 0 ]; then
	cmake -B ../build/linux/release/project -S . -G "Unix Makefiles" \
		-DUSE_DEBUG=OFF \
		-DCMAKE_BUILD_TYPE=RelWithDebInfo \
		-DBUILD_STATIC_LIB=$make_static_libs \
		$cmake_cuda_options \
		-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=../bin \
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../bin \
		-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=../lib
	exit_code=$?
fi
cd $saved_cwd

exit $exit_code
