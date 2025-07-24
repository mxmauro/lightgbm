package lightgbm

import (
	"fmt"
	"unsafe"
)

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdlib.h>
*/
import "C"

// -----------------------------------------------------------------------------

func loadLib(path string) error {
	var err error

	// Load library
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	handle := C.dlopen(cpath, C.RTLD_LAZY)
	if handle == nil {
		return fmt.Errorf("dlopen failed: %s", C.GoString(C.dlerror()))
	}

	// Load functions
	getProc := func(name string) unsafe.Pointer {
		cs := C.CString(name)
		defer C.free(unsafe.Pointer(cs))

		sym := C.dlsym(handle, cs)
		if sym == nil {
			if err == nil {
				err = fmt.Errorf("Missing symbol: %s", name)
			}
		}
		return sym, nil
	}

	savePointers(
		getProc("LGBM_GetLastError"),
		getProc("LGBM_RegisterLogCallback"),

		getProc("LGBM_DatasetCreateFromMat"),
		getProc("LGBM_DatasetFree"),
		getProc("LGBM_DatasetSetField"),

		getProc("LGBM_BoosterCreate"),
		getProc("LGBM_BoosterFree"),
		getProc("LGBM_BoosterAddValidData"),
		getProc("LGBM_BoosterUpdateOneIter"),
		getProc("LGBM_BoosterGetEval"),
		getProc("LGBM_BoosterGetEvalCounts"),
		getProc("LGBM_BoosterGetNumFeature"),
		getProc("LGBM_BoosterGetNumClasses"),
		getProc("LGBM_BoosterGetNumPredict"),
		getProc("LGBM_BoosterSaveModelToString"),
		getProc("LGBM_BoosterLoadModelFromString"),

		getProc("LGBM_BoosterPredictForMatSingleRowFastInit"),
		getProc("LGBM_BoosterPredictForMatSingleRowFast"),
		getProc("LGBM_FastConfigFree"),
	)

	// Done
	return err
}
