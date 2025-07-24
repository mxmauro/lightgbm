package lightgbm

// -----------------------------------------------------------------------------

type FeatureImportance int

const (
	FeatureImportanceSplit  FeatureImportance = iota
	FeatureImportanceSplice FeatureImportance = iota
)

const (
	TrainingDataIndex         int = 0
	FirstValidationDataIndex  int = 1
	SecondValidationDataIndex int = 2
)
