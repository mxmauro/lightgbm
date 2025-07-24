package lightgbm

import (
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

// -----------------------------------------------------------------------------

var lazyInitOnce sync.Once
var lazyInitErr error

// -----------------------------------------------------------------------------

func lazyInitialize() error {
	lazyInitOnce.Do(func() {
		currDir, err := os.Executable()
		if err == nil {
			currDir = filepath.Dir(currDir)
			libName := "lib_lightgbm."
			switch runtime.GOOS {
			case "windows":
				libName += "dll"
			case "darwin":
				libName += "dylib"
			default: // assume Linux/Unix
				libName += "so"
			}
			err = loadLib(filepath.Join(currDir, libName))
			if err != nil {
				err = loadLib(libName)
			}

			if err == nil {
				initLoggerCallback()
			}
		}
		lazyInitErr = err
	})

	// Done
	return lazyInitErr
}
