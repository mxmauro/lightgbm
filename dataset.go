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
	refDS             *Dataset
	parameters        string
	ptr               unsafe.Pointer
	features          []float64
	featuresCount     int
	featuresRowsCount int
	featureNames      []string
	groups            []int32
	groupsCount       int
	labels            []float32
	labelsCount       int
	initScores        []float64
	initScoresCount   int
	weights           []float32
	weightsCount      int
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
	ds.featuresRowsCount += 1

	// Done
	return nil
}

func (ds *Dataset) SetFeatureNames(names []string) error {
	if ds.ptr != nil {
		return errors.New("dataset cannot be modified")
	}

	// Copy names
	if len(names) > 0 {
		ds.featureNames = make([]string, len(names))
		copy(ds.featureNames, names)
	} else {
		ds.featureNames = nil
	}

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

func (ds *Dataset) SetWeight(data float64) error {
	return ds.SetWeights([]float64{data})
}

func (ds *Dataset) SetWeights(data []float64) error {
	if ds.ptr != nil {
		return errors.New("dataset cannot be expanded")
	}

	// First row?
	if len(ds.weights) == 0 {
		// Check data length
		if len(data) == 0 {
			return errors.New("empty data")
		}
		ds.weightsCount = len(data)
		ds.weights = make([]float32, 0, ds.weightsCount*expansionMultiplier)
	} else {
		// Check data length
		if len(data) != ds.weightsCount {
			return errors.New("rows of data must contain the same amount of weights")
		}

		if len(ds.weights)+ds.weightsCount > cap(ds.weights) {
			expandedWeights := make([]float32, len(ds.weights), cap(ds.weights)+ds.weightsCount*expansionMultiplier)
			copy(expandedWeights, ds.weights)
			ds.weights = expandedWeights
		}
	}

	ofs := len(ds.weights)
	ds.weights = ds.weights[:ofs+len(data)]
	for idx := 0; idx < len(data); idx++ {
		ds.weights[ofs+idx] = float32(data[idx])
	}

	// Done
	return nil
}

func (ds *Dataset) SetInitScore(data float64) error {
	return ds.SetInitScores([]float64{data})
}

func (ds *Dataset) SetInitScores(data []float64) error {
	if ds.ptr != nil {
		return errors.New("dataset cannot be expanded")
	}

	// First row?
	if len(ds.initScores) == 0 {
		// Check data length
		if len(data) == 0 {
			return errors.New("empty data")
		}
		ds.initScoresCount = len(data)
		ds.initScores = make([]float64, 0, ds.initScoresCount*expansionMultiplier)
	} else {
		// Check data length
		if len(data) != ds.initScoresCount {
			return errors.New("rows of data must contain the same amount of init scores")
		}

		if len(ds.initScores)+ds.initScoresCount > cap(ds.initScores) {
			expandedInitScores := make([]float64, len(ds.initScores), cap(ds.initScores)+ds.initScoresCount*expansionMultiplier)
			copy(expandedInitScores, ds.initScores)
			ds.initScores = expandedInitScores
		}
	}

	ofs := len(ds.initScores)
	ds.initScores = ds.initScores[:ofs+len(data)]
	copy(ds.initScores[ofs:], data)

	// Done
	return nil
}

func (ds *Dataset) SetGroup(grp int) error {
	return ds.SetGroups([]int{grp})
}

func (ds *Dataset) SetGroups(data []int) error {
	if ds.ptr != nil {
		return errors.New("dataset cannot be expanded")
	}

	// First row?
	if len(ds.groups) == 0 {
		// Check data length
		if len(data) == 0 {
			return errors.New("empty data")
		}
		ds.groupsCount = len(data)
		ds.groups = make([]int32, 0, ds.groupsCount*expansionMultiplier)
	} else {
		// Check data length
		if len(data) != ds.groupsCount {
			return errors.New("rows of data must contain the same amount of groups")
		}

		if len(ds.groups)+ds.groupsCount > cap(ds.groups) {
			expandedGroups := make([]int32, len(ds.groups), cap(ds.groups)+ds.groupsCount*expansionMultiplier)
			copy(expandedGroups, ds.groups)
			ds.groups = expandedGroups
		}
	}

	ofs := len(ds.groups)
	ds.groups = ds.groups[:ofs+len(data)]
	for idx := 0; idx < len(data); idx++ {
		ds.groups[ofs+idx] = int32(data[idx])
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
	datasetPtr, err := datasetCreateFromMat(ds.features, ds.featuresRowsCount, ds.parameters, ref)
	if err != nil {
		return nil, err
	}

	if ds.featureNames != nil {
		featuresCount := len(ds.features) / ds.featuresRowsCount
		if len(ds.featureNames) != featuresCount {
			datasetFree(datasetPtr)
			return nil, errors.New("the number of feature columns does not match the number of names")
		}
		err = datasetSetFeatureNames(datasetPtr, ds.featureNames)
		if err != nil {
			datasetFree(datasetPtr)
			return nil, err
		}
	}

	if ds.labels != nil {
		err = datasetSetFieldFloat32(datasetPtr, "label", ds.labels)
		if err != nil {
			datasetFree(datasetPtr)
			return nil, err
		}
	}
	if ds.weights != nil {
		err = datasetSetFieldFloat32(datasetPtr, "weight", ds.weights)
		if err != nil {
			datasetFree(datasetPtr)
			return nil, err
		}
	}
	if ds.initScores != nil {
		err = datasetSetFieldFloat64(datasetPtr, "init_score", ds.initScores)
		if err != nil {
			datasetFree(datasetPtr)
			return nil, err
		}
	}
	if ds.groups != nil {
		err = datasetSetFieldInt32(datasetPtr, "group", ds.groups)
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
