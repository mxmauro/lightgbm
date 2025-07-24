package lightgbm

import (
	"runtime"
	"strings"
	"unsafe"
)

// -----------------------------------------------------------------------------

type Booster struct {
	ptr              unsafe.Pointer
	validDatasetList []*Dataset
}

// -----------------------------------------------------------------------------

func NewBoosterFromDataset(ds *Dataset, parameters []string, validators []*Dataset) (*Booster, error) {
	var boosterPtr unsafe.Pointer

	if ds == nil {
		return nil, ErrNotInitialized
	}
	for _, validator := range validators {
		if validator == nil {
			return nil, ErrNotInitialized
		}
	}

	// Get the dataset handle
	datasetPtr, err := ds.getPtr()
	if err != nil {
		return nil, err
	}

	// Create the booster object
	boosterPtr, err = boosterCreate(datasetPtr, strings.Join(parameters, " "))
	if err != nil {
		return nil, err
	}

	// Add validators if they were specified
	for _, validator := range validators {
		var validatorDatasetPtr unsafe.Pointer

		validatorDatasetPtr, err = validator.getPtr()
		if err != nil {
			boosterFree(boosterPtr)
			return nil, err
		}
		err = boosterAddValidData(boosterPtr, validatorDatasetPtr)
		if err != nil {
			boosterFree(boosterPtr)
			return nil, err
		}
	}

	// Create the booster object
	b := &Booster{
		ptr:              boosterPtr,
		validDatasetList: validators,
	}
	runtime.SetFinalizer(b, func(b *Booster) {
		b.finalize()
	})

	// Done
	return b, nil
}

func NewBoosterFromString(data string) (*Booster, error) {
	boosterPtr, err := boosterLoadModelFromString(data)
	if err != nil {
		return nil, err
	}

	// Create the booster object
	b := &Booster{
		ptr: boosterPtr,
	}
	runtime.SetFinalizer(b, func(b *Booster) {
		b.finalize()
	})

	// Done
	return b, nil
}

func (b *Booster) UpdateOneIter() (bool, error) {
	return boosterUpdateOneIter(b.ptr)
}

func (b *Booster) GetEval(dataIdx int) ([]float64, error) {
	return boosterGetEval(b.ptr, dataIdx)
}

func (b *Booster) ToString(featureImportance FeatureImportance) (string, error) {
	return boosterSaveModelToString(b.ptr, int(featureImportance))
}
func (b *Booster) Predictor(rawScore bool, parameters []string) (*Predictor, error) {
	return NewPredictorFromBooster(b, rawScore, parameters)
}

func (b *Booster) finalize() {
	boosterFree(b.ptr)
	b.validDatasetList = nil
}
