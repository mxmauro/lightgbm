package lightgbm

import "C"
import (
	"errors"
	"runtime"
	"strings"
	"unsafe"
)

// -----------------------------------------------------------------------------

type Predictor struct {
	ptr           unsafe.Pointer
	b             *Booster
	featuresCount int
	classesCount  int
}

// -----------------------------------------------------------------------------

func NewPredictorFromBooster(b *Booster, rawScore bool, parameters []string) (*Predictor, error) {
	var classesCount int
	var ptr unsafe.Pointer

	if b == nil {
		return nil, ErrNotInitialized
	}

	// Get the number of features in the booster object
	featuresCount, err := boosterGetFeaturesCount(b.ptr)
	if err != nil {
		return nil, err
	}

	// Get the number of features in the booster object
	classesCount, err = boosterGetClassesCount(b.ptr)
	if err != nil {
		return nil, err
	}

	// Create the predictor object
	ptr, err = boosterPredictForMatSingleRowFastInit(b.ptr, rawScore, strings.Join(parameters, " "))
	if err != nil {
		return nil, err
	}

	// Create the fast predictor object
	p := &Predictor{
		ptr:           ptr,
		b:             b,
		featuresCount: featuresCount,
		classesCount:  classesCount,
	}
	runtime.SetFinalizer(p, func(p *Predictor) {
		p.finalize()
	})

	// Done
	return p, nil
}

func (p *Predictor) Predict(features []float64) ([]float64, error) {
	if len(features) != p.featuresCount {
		return nil, errors.New("feature count does not match number of features in model")
	}

	// Create output
	out := make([]float64, p.classesCount)

	// Predict
	err := boosterPredictForMatSingleRowFast(p.ptr, features, out)
	if err != nil {
		return nil, err
	}

	// Done
	return out, nil
}

func (p *Predictor) finalize() {
	predictFastConfigFree(p.ptr)
	p.b = nil
}
