package lightgbm

import (
	"errors"
	"runtime"
	"strings"
	"unsafe"
)

// -----------------------------------------------------------------------------

const (
	expansionMultiplier = 10240
)

// -----------------------------------------------------------------------------

type Dataset struct {
	refDS         *Dataset
	parameters    string
	ptr           unsafe.Pointer
	features      []float64
	featuresCount int
	labels        []float32
	labelsCount   int
}

// -----------------------------------------------------------------------------

func NewDataset(parameters []string) *Dataset {
	return NewDatasetWithReference(parameters, nil)
}

func NewDatasetWithReference(parameters []string, refDS *Dataset) *Dataset {
	// Create the dataset object
	ds := &Dataset{
		refDS:      refDS,
		parameters: strings.Join(parameters, " "),
	}
	runtime.SetFinalizer(ds, func(ds *Dataset) {
		ds.finalize()
	})

	// Done
	return ds
}

func (ds *Dataset) AddFeatureData(data []float64) error {
	if ds.ptr != nil {
		return errors.New("dataset cannot be expanded")
	}

	// First row?
	if len(ds.features) == 0 {
		// Check data length
		if len(data) == 0 {
			return errors.New("empty data")
		}
		ds.featuresCount = len(data)
		ds.features = make([]float64, 0, ds.featuresCount*expansionMultiplier)
	} else {
		// Check data length
		if len(data) != ds.featuresCount {
			return errors.New("rows of data must contain the same amount of features")
		}

		if len(ds.features)+ds.featuresCount > cap(ds.features) {
			expandedFeatures := make([]float64, len(ds.features), cap(ds.features)+ds.featuresCount*expansionMultiplier)
			copy(expandedFeatures, ds.features)
			ds.features = expandedFeatures
		}

	}

	ds.features = append(ds.features, data...)

	// Done
	return nil
}

func (ds *Dataset) SetLabel(data float64) error {
	return ds.SetLabels([]float64{data})
}

func (ds *Dataset) SetLabels(data []float64) error {
	if ds.ptr != nil {
		return errors.New("dataset cannot be expanded")
	}

	// First row?
	if len(ds.labels) == 0 {
		// Check data length
		if len(data) == 0 {
			return errors.New("empty data")
		}
		ds.labelsCount = len(data)
		ds.labels = make([]float32, 0, ds.labelsCount*expansionMultiplier)
	} else {
		// Check data length
		if len(data) != ds.labelsCount {
			return errors.New("rows of data must contain the same amount of labels")
		}

		if len(ds.labels)+ds.labelsCount > cap(ds.labels) {
			expandedLabels := make([]float32, len(ds.labels), cap(ds.labels)+ds.labelsCount*expansionMultiplier)
			copy(expandedLabels, ds.labels)
			ds.labels = expandedLabels
		}
	}

	ofs := len(ds.labels)
	ds.labels = ds.labels[:ofs+len(data)]
	for idx := 0; idx < len(data); idx++ {
		ds.labels[ofs+idx] = float32(data[idx])
	}

	// Done
	return nil

}

func (ds *Dataset) getPtr() (unsafe.Pointer, error) {
	var ref unsafe.Pointer

	if ds.ptr != nil {
		return ds.ptr, nil // Already created
	}
	if len(ds.features) == 0 {
		return nil, errors.New("dataset has no features")
	}

	// Create dataset
	if ds.refDS != nil {
		ref = ds.refDS.ptr
	}
	datasetPtr, err := datasetCreateFromMat(ds.features, ds.featuresCount, ds.parameters, ref)
	if err != nil {
		return nil, err
	}

	//  label, weight, init_score, group
	if ds.labels != nil {
		err = datasetSetFieldFloat32(datasetPtr, "label", ds.labels)
		if err != nil {
			datasetFree(datasetPtr)
			return nil, err
		}
	}

	// Assign pointer
	ds.ptr = datasetPtr

	// Done
	return datasetPtr, nil
}

func (ds *Dataset) finalize() {
	if ds.ptr != nil {
		datasetFree(ds.ptr)
		ds.ptr = nil
	}
}
