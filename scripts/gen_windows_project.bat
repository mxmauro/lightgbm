@ECHO OFF
SETLOCAL enabledelayedexpansion enableextensions 

IF "%VSINSTALLDIR%" == "" (
	ECHO [ERROR] This script must be run from a Visual Studio Developer Command Prompt.
	EXIT /b 1
)

REM Get the compiler version
SET _CL_VER=
FOR /f "tokens=7" %%a IN ('cl 2^>^&1 ^| FINDSTR /b /c:"Microsoft (R) C/C++"') DO (
	SET "_CL_VER=%%a"
)
IF "!_CL_VER!" == "" (
	ECHO Error: MSVC compiler not found in PATH.
	EXIT /b 1
)

REM Extract major.minor version
SET _CL_MAJOR=
SET _CL_MINOR=
FOR /f "tokens=1,2 delims=." %%i IN ("!_CL_VER!") DO (
	SET "_CL_MAJOR=%%i"
	SET "_CL_MINOR=%%j"
)

REM Determine toolset from version
IF "!_CL_MAJOR!" == "19" (
	IF "!_CL_MINOR!" geq "30" (
		SET _MSVC_VERSION=14.3
	) ELSE IF "!_CL_MINOR!" geq "20" (
		SET _MSVC_VERSION=14.2
	) ELSE IF "!_CL_MINOR!" geq "10" (
		SET _MSVC_VERSION=14.1
	) ELSE (
		ECHO Error: Unknown or unsupported compiler version: !_CL_VER!
		EXIT /b 1
	)
) ELSE (
	ECHO Error: Unknown or unsupported compiler version: !_CL_VER!
	EXIT /b 1
)

REM Parse command line arguments
SET _BOOST_BASE_DIR=
SET _NVIDIA_CUDA_BASE_DIR=
SET _MAKE_STATIC_LIBS=OFF

:parse_args
IF "%~1" == "" GOTO parse_args_done

IF "%~1" == "--boost-dir" (
	IF "%~2" == "" (
		ECHO Error: --boost-dir requires a directory path.
		EXIT /b 1
	)
	SET "_BOOST_BASE_DIR=%~2"
	SHIFT /1
) ELSE IF "%~1" == "--nvidia-cuda-dir" (
	IF "%~2" == "" (
		ECHO Error: --nvidia-cuda-dir requires a directory path.
		EXIT /b 1
	)
	SET "_NVIDIA_CUDA_BASE_DIR=%~2"
	SHIFT /1
) ELSE IF "%~1" == "--make-static" (
	SET _MAKE_STATIC_LIBS=ON
) ELSE (
	ECHO Error: Invalid parameter '%~1'
	EXIT /b 1
)

SHIFT /1
GOTO parse_args
:parse_args_done

REM Check options
IF "!_NVIDIA_CUDA_BASE_DIR!" == "auto" (
	IF "%CUDA_PATH%" == "" (
		SET _NVIDIA_CUDA_BASE_DIR=
	) ELSE (
		ECHO Info: Detected CUDA in directory '%CUDA_PATH%'
		SET "_NVIDIA_CUDA_BASE_DIR=%CUDA_PATH%"
	)
)
IF "!_NVIDIA_CUDA_BASE_DIR!" == "" (
	SET _CMAKE_CUDA_OPTIONS=-DUSE_GPU=OFF
) ELSE (
	REM CUDA requires boost
	IF "!_BOOST_BASE_DIR!" == "" (
		ECHO Error: Missing --boost-dir argument.
		EXIT /b 1
	)

	ECHO Info: Adding support for CUDA.

	SET _CMAKE_CUDA_OPTIONS=-DUSE_GPU=ON ^
		-DBoost_NO_SYSTEM_PATHS=ON ^
		-DBoost_USE_STATIC_LIBS=ON ^
		-DBoost_USE_STATIC_RUNTIME=ON ^
		-DBOOST_ROOT="!_BOOST_BASE_DIR!" ^
		-DBOOST_LIBRARYDIR="!_BOOST_BASE_DIR!/lib64-msvc-!_MSVC_VERSION!" ^
		-DOpenCL_LIBRARY="!_NVIDIA_CUDA_BASE_DIR!/lib/x64/OpenCL.lib" ^
		-DOpenCL_INCLUDE_DIR="!_NVIDIA_CUDA_BASE_DIR!/include"
)

REM Build projects
PUSHD ..\src
RMDIR /S /Q ..\build\windows 2>NUL

cmake -B ..\build\windows\debug\project -S . -G "NMake Makefiles" ^
	-DUSE_DEBUG=ON ^
	-DCMAKE_BUILD_TYPE=Debug ^
	-DBUILD_STATIC_LIB=!_MAKE_STATIC_LIBS! ^
	!_CMAKE_CUDA_OPTIONS! ^
	-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=../bin ^
	-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../bin ^
	-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=../lib ^
	-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreadedDebug"
SET "_EXIT_CODE=!ERRORLEVEL!"
IF !_EXIT_CODE! == 0 (
	cmake -B ..\build\windows\release\project -S . -G "NMake Makefiles" ^
		-DUSE_DEBUG=OFF ^
		-DCMAKE_BUILD_TYPE=RelWithDebInfo ^
		-DBUILD_STATIC_LIB=!_MAKE_STATIC_LIBS! ^
		!_CMAKE_CUDA_OPTIONS! ^
		-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=../bin ^
		-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=../bin ^
		-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=../lib ^
		-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded"
	SET "_EXIT_CODE=!ERRORLEVEL!"
)
POPD

EXIT /b !_EXIT_CODE!
