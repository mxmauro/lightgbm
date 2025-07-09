@ECHO OFF
SETLOCAL enabledelayedexpansion

SET _LIB_VERSION=v4.6.0

PUSHD ..
RMDIR /S /Q src >NUL 2>NUL
MKDIR src
CD src
git clone --branch %_LIB_VERSION% --depth 1 https://github.com/microsoft/LightGBM .
SET "_EXIT_CODE=%ERRORLEVEL%"
IF %_EXIT_CODE% == 0 (
	git submodule update --init --recursive
	SET "_EXIT_CODE=%ERRORLEVEL%"
)
IF %_EXIT_CODE% == 0 (
	REM Remove boost from compute submodule because it gives troubles
	RMDIR /S /Q external_libs\compute\include\boost >NUL 2>NUL
)
POPD

EXIT /b %_EXIT_CODE%
