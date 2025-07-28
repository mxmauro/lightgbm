package lightgbm

import "C"
import (
	"syscall"
	"unsafe"
)

// -----------------------------------------------------------------------------

var dll *syscall.LazyDLL

// -----------------------------------------------------------------------------

func loadLib(path string) error {
	var err error

	// Load library
	dll = syscall.NewLazyDLL(path)

	// Load functions
	getProc := func(name string) unsafe.Pointer {
		proc := dll.NewProc(name)
		err2 := proc.Find()
		if err2 != nil {
			if err == nil {
				err = err2
			}
			return nil
		}
		return unsafe.Pointer(proc.Addr())
	}

	savePointers(
		getProc("LGBM_GetLastError"),
		getProc("LGBM_RegisterLogCallback"),

		getProc("LGBM_DatasetCreateFromMat"),
		getProc("LGBM_DatasetFree"),
		getProc("LGBM_DatasetSetField"),
		getProc("LGBM_DatasetSetFeatureNames"),

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
