package lightgbm_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/mxmauro/lightgbm"
)

// -----------------------------------------------------------------------------

type TestData struct {
	Features     [][]float64
	Labels       []float64
	FeatureNames []string
}

// -----------------------------------------------------------------------------

func TestClassification(t *testing.T) {
	testOverall(t, "classification")
}

func TestRegression(t *testing.T) {
	testOverall(t, "regression")
}

func testOverall(t *testing.T, taskType string) {
	var p *lightgbm.Predictor

	lightgbm.LoggerSetCallback(func(msgType string, msg string) {
		t.Log("["+msgType+"]:", msg)
	})

	trainData, testData := generateTestData(2000, 4, taskType, 0.3)

	t.Log("Creating training dataset")
	ds := lightgbm.NewDataset(nil)

	t.Log("Adding training data")
	for _, data := range trainData.Features {
		err := ds.AddFeatureData(data)
		if err != nil {
			t.Fatal(err)
		}
	}
	t.Log("Adding training labels")
	for _, label := range trainData.Labels {
		err := ds.SetLabel(label)
		if err != nil {
			t.Fatal(err)
		}
	}

	t.Log("Creating booster from dataset")
	var boosterParams []string
	if taskType == "regression" {
		boosterParams = []string{
			"objective=regression",
			"metric=rmse",
			"boosting_type=gbdt",
			"num_leaves=31",
			"learning_rate=0.1",
			"feature_fraction=0.9",
			"bagging_fraction=0.8",
			"bagging_freq=5",
			"min_child_samples=20",
			"force_col_wise=true",
			"verbosity=1",
		}
	} else {
		boosterParams = []string{
			"objective=binary",
			"metric=binary_logloss",
			"boosting_type=gbdt",
			"num_leaves=31",
			"learning_rate=0.1",
			"feature_fraction=0.9",
			"bagging_fraction=0.8",
			"bagging_freq=5",
			"min_child_samples=20",
			"is_unbalance=false",
			"force_col_wise=true",
			"verbosity=1",
		}
	}
	b, err := lightgbm.NewBoosterFromDataset(ds, boosterParams, nil)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("Updating booster")
	for i := 0; i < 100; i++ {
		var isFinished bool

		isFinished, err = b.UpdateOneIter()
		if err != nil {
			t.Fatal(err)
		}
		if isFinished {
			break
		}
	}

	t.Log("Creating predictor from booster")
	p, err = b.Predictor(false, nil)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("Predicting test data")
	good := 0
	bad := 0
	veryBad := 0

	minLabel := testData.Labels[0]
	maxLabel := testData.Labels[0]
	for idx := range testData.Labels {
		if testData.Labels[idx] < minLabel {
			minLabel = testData.Labels[idx]
		}
		if testData.Labels[idx] > maxLabel {
			maxLabel = testData.Labels[idx]
		}
	}
	diff10Pct := (maxLabel - minLabel) * 0.1
	diff40Pct := (maxLabel - minLabel) * 0.4
	for idx, data := range testData.Features {
		var predictions []float64

		predictions, err = p.Predict(data)
		if err != nil {
			t.Fatal(err)
		}
		if len(predictions) != 1 {
			t.Fatal("unexpected number of predictions")
		}

		//		t.Log("  -> Predicted:", predictions[0], "/ Expected:", testData.Labels[idx])
		if predictions[0] < testData.Labels[idx]-diff40Pct || predictions[0] > testData.Labels[idx]+diff40Pct {
			veryBad += 1
		} else if predictions[0] < testData.Labels[idx]-diff10Pct || predictions[0] > testData.Labels[idx]+diff10Pct {
			bad += 1
		} else {
			good += 1
		}
	}
	t.Log("Good:", good, "/ Bad:", bad, "/ Very bad:", veryBad)

	if float64(good)/float64(good+bad+veryBad) < 0.9 {
		t.FailNow()
	}
}

func generateTestData(samplesCount int, featuresCount int, taskType string, testRatio float64) (*TestData, *TestData) {
	data := TestData{
		Features:     make([][]float64, samplesCount),
		Labels:       make([]float64, samplesCount),
		FeatureNames: make([]string, featuresCount),
	}

	// Generate feature names
	for i := 0; i < featuresCount; i++ {
		data.FeatureNames[i] = fmt.Sprintf("feature_%d", i)
	}

	// Generate synthetic data
	for i := 0; i < samplesCount; i++ {
		data.Features[i] = make([]float64, featuresCount)

		// Generate features with some correlation structure
		for j := 0; j < featuresCount; j++ {
			// Mix of normal and uniform distributions
			if j%2 == 0 {
				data.Features[i][j] = rand.NormFloat64()*2 + 1
			} else {
				data.Features[i][j] = rand.Float64()*10 - 5
			}
		}

		// Generate labels based on the task type
		if taskType == "regression" {
			// Create a non-linear relationship for regression
			label := 0.0
			for j := 0; j < featuresCount; j++ {
				weight := 1.0 / float64(j+1) // Decreasing weights
				label += weight * data.Features[i][j]
				if j < 2 {
					label += 0.1 * data.Features[i][j] * data.Features[i][j] // Non-linear terms
				}
			}
			label += rand.NormFloat64() * 0.1 // Add noise
			data.Labels[i] = label
		} else { // binary classification
			// Create a decision boundary
			score := 0.0
			for j := 0; j < featuresCount; j++ {
				weight := math.Pow(-1, float64(j)) / float64(j+1)
				score += weight * data.Features[i][j]
			}

			// Add interaction terms
			if featuresCount >= 2 {
				score += 0.5 * data.Features[i][0] * data.Features[i][1]
			}

			// Convert to probability and then to binary label
			prob := 1.0 / (1.0 + math.Exp(-score))
			if prob > 0.5 {
				data.Labels[i] = 1.0
			} else {
				data.Labels[i] = 0.0
			}
		}
	}

	testCount := int(float64(samplesCount) * testRatio)
	trainCount := samplesCount - testCount

	// Create the train set
	train := &TestData{
		Features:     data.Features[:trainCount],
		Labels:       data.Labels[:trainCount],
		FeatureNames: data.FeatureNames,
	}

	// Create the test set
	test := &TestData{
		Features:     data.Features[trainCount:],
		Labels:       data.Labels[trainCount:],
		FeatureNames: data.FeatureNames,
	}

	// Done
	return train, test
}
