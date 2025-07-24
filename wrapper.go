package lightgbm

import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

/*
#include <stdlib.h>
#include <stdint.h>

// -----------------------------------------------------------------------------

#define C_API_DTYPE_FLOAT32 0
#define C_API_DTYPE_FLOAT64 1

#define C_API_FEATURE_IMPORTANCE_SPLIT 0
#define C_API_FEATURE_IMPORTANCE_GAIN  1

#define C_API_PREDICT_NORMAL     0
#define C_API_PREDICT_RAW_SCORE  1
#define C_API_PREDICT_LEAF_INDEX 2
#define C_API_PREDICT_CONTRIB    3

// -----------------------------------------------------------------------------

typedef void* DatasetHandle;
typedef void* BoosterHandle;
typedef void* FastConfigHandle;

typedef void (*lpfnLogCallback)(const char*);

// -----------------------------------------------------------------------------

typedef char* (*lpfnLGBM_GetLastError)();

typedef int (*lpfnLGBM_RegisterLogCallback)(lpfnLogCallback callback);

typedef int (*lpfnLGBM_DatasetCreateFromMat)(const void* data,
                                             int data_type,
                                             int32_t nrow,
                                             int32_t ncol,
                                             int is_row_major,
                                             const char* parameters,
                                             const DatasetHandle reference,
                                             DatasetHandle* out);
typedef int (*lpfnLGBM_DatasetFree)(DatasetHandle handle);

typedef int (*lpfnLGBM_DatasetSetField)(DatasetHandle handle,
                                        const char* field_name,
                                        const void* field_data,
                                        int num_element,
                                        int type);

typedef int (*lpfnLGBM_BoosterCreate)(const DatasetHandle train_data,
                                      const char* parameters,
                                      BoosterHandle* out);

typedef int (*lpfnLGBM_BoosterFree)(BoosterHandle handle);

typedef int (*lpfnLGBM_BoosterAddValidData)(BoosterHandle handle,
                                            const DatasetHandle valid_data);

typedef int (*lpfnLGBM_BoosterUpdateOneIter)(BoosterHandle handle,
                                             int* is_finished);

typedef int (*lpfnLGBM_BoosterGetEval)(BoosterHandle handle,
                                       int data_idx,
                                       int* out_len,
                                       double* out_results);

typedef int (*lpfnLGBM_BoosterGetEvalCounts)(BoosterHandle handle,
                                             int* out_len);

typedef int (*lpfnLGBM_BoosterGetNumFeature)(BoosterHandle handle,
                                             int *out_len);

typedef int (*lpfnLGBM_BoosterGetNumClasses)(BoosterHandle handle,
                                             int *out_len);

typedef int (*lpfnLGBM_BoosterGetNumPredict)(BoosterHandle handle,
                                             int data_idx,
                                             int64_t *out_len);

typedef int (*lpfnLGBM_BoosterSaveModelToString)(BoosterHandle handle,
                                                 int start_iteration,
                                                 int num_iteration,
                                                 int feature_importance_type,
                                                 int64_t buffer_len,
                                                 int64_t* out_len,
                                                 char* out_str);

typedef int (*lpfnLGBM_BoosterLoadModelFromString)(const char* model_str,
                                                   int* out_num_iterations,
                                                   BoosterHandle* out);

typedef int (*lpfnLGBM_BoosterPredictForMatSingleRowFastInit)(BoosterHandle handle,
                                                              const int predict_type,
                                                              const int start_iteration,
                                                              const int num_iteration,
                                                              const int data_type,
                                                              const int32_t ncol,
                                                              const char *parameter,
                                                              FastConfigHandle *out_fastConfig);

typedef int (*lpfnLGBM_BoosterPredictForMatSingleRowFast)(FastConfigHandle fastConfig_handle,
                                                          const void *data,
                                                          int64_t *out_len,
                                                          double *out_result);

typedef int (*lpfnLGBM_FastConfigFree)(FastConfigHandle fastConfig);

// -----------------------------------------------------------------------------

static lpfnLGBM_GetLastError fnLGBM_GetLastError = nullptr;
static lpfnLGBM_RegisterLogCallback fnLGBM_RegisterLogCallback = nullptr;

static lpfnLGBM_DatasetCreateFromMat fnLGBM_DatasetCreateFromMat = nullptr;
static lpfnLGBM_DatasetFree          fnLGBM_DatasetFree          = nullptr;
static lpfnLGBM_DatasetSetField      fnLGBM_DatasetSetField      = nullptr;

static lpfnLGBM_BoosterCreate              fnLGBM_BoosterCreate              = nullptr;
static lpfnLGBM_BoosterFree                fnLGBM_BoosterFree                = nullptr;
static lpfnLGBM_BoosterAddValidData        fnLGBM_BoosterAddValidData        = nullptr;
static lpfnLGBM_BoosterUpdateOneIter       fnLGBM_BoosterUpdateOneIter       = nullptr;
static lpfnLGBM_BoosterGetEval             fnLGBM_BoosterGetEval             = nullptr;
static lpfnLGBM_BoosterGetEvalCounts       fnLGBM_BoosterGetEvalCounts       = nullptr;
static lpfnLGBM_BoosterGetNumFeature       fnLGBM_BoosterGetNumFeature       = nullptr;
static lpfnLGBM_BoosterGetNumClasses       fnLGBM_BoosterGetNumClasses       = nullptr;
static lpfnLGBM_BoosterGetNumPredict       fnLGBM_BoosterGetNumPredict       = nullptr;
static lpfnLGBM_BoosterSaveModelToString   fnLGBM_BoosterSaveModelToString   = nullptr;
static lpfnLGBM_BoosterLoadModelFromString fnLGBM_BoosterLoadModelFromString = nullptr;

static lpfnLGBM_BoosterPredictForMatSingleRowFastInit fnLGBM_BoosterPredictForMatSingleRowFastInit = nullptr;
static lpfnLGBM_BoosterPredictForMatSingleRowFast     fnLGBM_BoosterPredictForMatSingleRowFast     = nullptr;
static lpfnLGBM_FastConfigFree                        fnLGBM_FastConfigFree                        = nullptr;

// -----------------------------------------------------------------------------

static void savePointers(void *ptr_LGBM_GetLastError,
                         void *ptr_LGBM_RegisterLogCallback,
                         void *ptr_LGBM_DatasetCreateFromMat,
                         void *ptr_LGBM_DatasetFree,
                         void *ptr_LGBM_DatasetSetField,
                         void *ptr_LGBM_BoosterCreate,
                         void *ptr_LGBM_BoosterFree,
                         void *ptr_LGBM_BoosterAddValidData,
                         void *ptr_LGBM_BoosterUpdateOneIter,
                         void *ptr_LGBM_BoosterGetEval,
                         void *ptr_LGBM_BoosterGetEvalCounts,
                         void *ptr_LGBM_BoosterGetNumFeature,
                         void *ptr_LGBM_BoosterGetNumClasses,
                         void *ptr_LGBM_BoosterGetNumPredict,
                         void *ptr_LGBM_BoosterSaveModelToString,
                         void *ptr_LGBM_BoosterLoadModelFromString,
                         void *ptr_LGBM_BoosterPredictForMatSingleRowFastInit,
                         void *ptr_LGBM_BoosterPredictForMatSingleRowFast,
                         void *ptr_LGBM_FastConfigFree)
{
    fnLGBM_GetLastError = (lpfnLGBM_GetLastError)ptr_LGBM_GetLastError;
    fnLGBM_RegisterLogCallback = (lpfnLGBM_RegisterLogCallback)ptr_LGBM_RegisterLogCallback;

    fnLGBM_DatasetCreateFromMat = (lpfnLGBM_DatasetCreateFromMat)ptr_LGBM_DatasetCreateFromMat;
    fnLGBM_DatasetFree          = (lpfnLGBM_DatasetFree         )ptr_LGBM_DatasetFree;
    fnLGBM_DatasetSetField      = (lpfnLGBM_DatasetSetField     )ptr_LGBM_DatasetSetField;

    fnLGBM_BoosterCreate              = (lpfnLGBM_BoosterCreate             )ptr_LGBM_BoosterCreate;
    fnLGBM_BoosterFree                = (lpfnLGBM_BoosterFree               )ptr_LGBM_BoosterFree;
    fnLGBM_BoosterAddValidData        = (lpfnLGBM_BoosterAddValidData       )ptr_LGBM_BoosterAddValidData;
    fnLGBM_BoosterUpdateOneIter       = (lpfnLGBM_BoosterUpdateOneIter      )ptr_LGBM_BoosterUpdateOneIter;
    fnLGBM_BoosterGetEval             = (lpfnLGBM_BoosterGetEval            )ptr_LGBM_BoosterGetEval;
    fnLGBM_BoosterGetEvalCounts       = (lpfnLGBM_BoosterGetEvalCounts      )ptr_LGBM_BoosterGetEvalCounts;
    fnLGBM_BoosterGetNumFeature       = (lpfnLGBM_BoosterGetNumFeature      )ptr_LGBM_BoosterGetNumFeature;
    fnLGBM_BoosterGetNumClasses       = (lpfnLGBM_BoosterGetNumClasses      )ptr_LGBM_BoosterGetNumClasses;
    fnLGBM_BoosterGetNumPredict       = (lpfnLGBM_BoosterGetNumPredict      )ptr_LGBM_BoosterGetNumPredict;
    fnLGBM_BoosterSaveModelToString   = (lpfnLGBM_BoosterSaveModelToString  )ptr_LGBM_BoosterSaveModelToString;
    fnLGBM_BoosterLoadModelFromString = (lpfnLGBM_BoosterLoadModelFromString)ptr_LGBM_BoosterLoadModelFromString;

    fnLGBM_BoosterPredictForMatSingleRowFastInit = (lpfnLGBM_BoosterPredictForMatSingleRowFastInit)ptr_LGBM_BoosterPredictForMatSingleRowFastInit;
    fnLGBM_BoosterPredictForMatSingleRowFast     = (lpfnLGBM_BoosterPredictForMatSingleRowFast    )ptr_LGBM_BoosterPredictForMatSingleRowFast;
    fnLGBM_FastConfigFree                        = (lpfnLGBM_FastConfigFree                       )ptr_LGBM_FastConfigFree;
}

static char* call_LGBM_GetLastError()
{
    return fnLGBM_GetLastError();
}

static int call_LGBM_DatasetCreateFromMat(const void* data,
                                          int data_type,
                                          int32_t nrow,
                                          int32_t ncol,
                                          int is_row_major,
                                          const char* parameters,
                                          const DatasetHandle reference,
                                          DatasetHandle* out)
{
    return fnLGBM_DatasetCreateFromMat(data, data_type, nrow, ncol, is_row_major, parameters, reference, out);
}

static int call_LGBM_DatasetFree(DatasetHandle handle)
{
    return fnLGBM_DatasetFree(handle);
}

static int call_LGBM_DatasetSetField(DatasetHandle handle,
                                     const char* field_name,
                                     const void* field_data,
                                     int num_element,
                                     int type)
{
    return fnLGBM_DatasetSetField(handle, field_name, field_data, num_element, type);
}

static int call_LGBM_BoosterCreate(const DatasetHandle train_data,
                                   const char* parameters,
                                   BoosterHandle* out)
{
    return fnLGBM_BoosterCreate(train_data, parameters, out);
}

static int call_LGBM_BoosterFree(BoosterHandle handle)
{
    return fnLGBM_BoosterFree(handle);
}

static int call_LGBM_BoosterAddValidData(BoosterHandle handle,
                                         const DatasetHandle valid_data)
{
    return fnLGBM_BoosterAddValidData(handle, valid_data);
}

static int call_LGBM_BoosterUpdateOneIter(BoosterHandle handle,
                                          int* is_finished)
{
    return fnLGBM_BoosterUpdateOneIter(handle, is_finished);
}

static int call_LGBM_BoosterGetEval(BoosterHandle handle,
                                    int data_idx,
                                    int* out_len,
                                    double* out_results)
{
    return fnLGBM_BoosterGetEval(handle, data_idx, out_len, out_results);
}

static int call_LGBM_BoosterGetEvalCounts(BoosterHandle handle,
                                          int *out_len)
{
    return fnLGBM_BoosterGetEvalCounts(handle, out_len);
}

static int call_LGBM_BoosterGetNumClasses(BoosterHandle handle,
                                          int *out_len)
{
    return fnLGBM_BoosterGetNumClasses(handle, out_len);
}

static int call_LGBM_BoosterGetNumFeature(BoosterHandle handle,
                                          int *out_len)
{
    return fnLGBM_BoosterGetNumFeature(handle, out_len);
}

static int call_LGBM_BoosterSaveModelToString(BoosterHandle handle,
                                              int start_iteration,
                                              int num_iteration,
                                              int feature_importance_type,
                                              int64_t buffer_len,
                                              int64_t* out_len,
                                              char* out_str)
{
    return fnLGBM_BoosterSaveModelToString(handle, start_iteration, num_iteration, feature_importance_type,
                                           buffer_len, out_len, out_str);
}

static int call_LGBM_BoosterLoadModelFromString(const char* model_str,
                                                int* out_num_iterations,
                                                BoosterHandle* out)
{
    return fnLGBM_BoosterLoadModelFromString(model_str, out_num_iterations, out);
}


static int call_LGBM_BoosterPredictForMatSingleRowFastInit(BoosterHandle handle,
                                                           const int predict_type,
                                                           const int start_iteration,
                                                           const int num_iteration,
                                                           const int data_type,
                                                           const int32_t ncol,
                                                           const char *parameter,
                                                           FastConfigHandle *out_fastConfig)
{
    return fnLGBM_BoosterPredictForMatSingleRowFastInit(handle, predict_type, start_iteration, num_iteration,
                                                        data_type, ncol, parameter, out_fastConfig);
}

static int call_LGBM_BoosterPredictForMatSingleRowFast(FastConfigHandle fastConfig_handle,
                                                       const void *data,
                                                       int64_t *out_len,
                                                       double *out_result)
{
    return fnLGBM_BoosterPredictForMatSingleRowFast(fastConfig_handle, data, out_len, out_result);
}

static int call_LGBM_FastConfigFree(FastConfigHandle handle)
{
    return fnLGBM_FastConfigFree(handle);
}

extern void goLoggerCallback(char*);

static void initLoggerCallback()
{
    fnLGBM_RegisterLogCallback((lpfnLogCallback)goLoggerCallback);
}
*/
import "C"

// -----------------------------------------------------------------------------

var errInvalidHandle = errors.New("invalid handle")

var loggerCh chan string

func initLoggerCallback() {
	loggerCh = make(chan string, 4)
	go loggerCollector()

	C.initLoggerCallback()
}

//export goLoggerCallback
func goLoggerCallback(cMsg *C.char) {
	msg := C.GoString(cMsg)
	select {
	case loggerCh <- msg:
	default:
		// Channel is full, drop message
	}
}

func datasetCreateFromMat(features []float64, featuresCount int, parameters string, refHandle unsafe.Pointer) (unsafe.Pointer, error) {
	var handle unsafe.Pointer

	if len(features) == 0 || featuresCount <= 0 {
		return nil, errors.New("features count is zero")
	}

	rowsCount := len(features) / featuresCount

	if len(features) != featuresCount*rowsCount {
		return nil, errors.New("feature is not a matrix")
	}

	// Initialize engine
	if err := lazyInitialize(); err != nil {
		return nil, err
	}

	// Convert parameters
	cParams := C.CString(parameters)
	defer C.free(unsafe.Pointer(cParams))

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Create the dataset object
	ret := C.call_LGBM_DatasetCreateFromMat(
		unsafe.Pointer(&features[0]),
		C.int(C.C_API_DTYPE_FLOAT64),
		C.int32_t(rowsCount),
		C.int32_t(featuresCount),
		C.int32_t(1),
		cParams,
		C.DatasetHandle(refHandle),
		(*C.DatasetHandle)(&handle),
	)
	if ret != 0 {
		return nil, getLastError()
	}

	// Done
	return handle, nil
}

func datasetFree(handle unsafe.Pointer) {
	if handle != nil {
		_ = C.call_LGBM_DatasetFree(
			C.DatasetHandle(handle),
		)
	}
}

func datasetSetFieldFloat64(handle unsafe.Pointer, field string, values []float64) error {
	if handle == nil {
		return errInvalidHandle
	}

	// Convert parameters
	cFieldName := C.CString(field)
	defer C.free(unsafe.Pointer(cFieldName))

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Set field
	ret := C.call_LGBM_DatasetSetField(
		C.DatasetHandle(handle),
		cFieldName,
		unsafe.Pointer(&values[0]),
		C.int(len(values)),
		C.int(C.C_API_DTYPE_FLOAT64),
	)
	runtime.KeepAlive(values) // Yes, keep-alive should be placed after the position where is used
	if ret != 0 {
		return getLastError()
	}

	// Done
	return nil
}

func datasetSetFieldFloat32(handle unsafe.Pointer, field string, values []float32) error {
	if handle == nil {
		return errInvalidHandle
	}

	// Convert parameters
	cFieldName := C.CString(field)
	defer C.free(unsafe.Pointer(cFieldName))

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Set field
	ret := C.call_LGBM_DatasetSetField(
		C.DatasetHandle(handle),
		cFieldName,
		unsafe.Pointer(&values[0]),
		C.int(len(values)),
		C.int(C.C_API_DTYPE_FLOAT32),
	)
	runtime.KeepAlive(values) // Yes, keep-alive should be placed after the position where is used
	if ret != 0 {
		return getLastError()
	}

	// Done
	return nil
}

func boosterCreate(datasetHandle unsafe.Pointer, parameters string) (unsafe.Pointer, error) {
	var handle unsafe.Pointer

	if datasetHandle == nil {
		return nil, errInvalidHandle
	}

	// Convert parameters
	cParams := C.CString(parameters)
	defer C.free(unsafe.Pointer(cParams))

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Create the booster object
	ret := C.call_LGBM_BoosterCreate(
		C.DatasetHandle(datasetHandle),
		cParams,
		(*C.BoosterHandle)(&handle),
	)
	if ret != 0 {
		return nil, getLastError()
	}

	// Done
	return handle, nil
}

func boosterFree(handle unsafe.Pointer) {
	if handle != nil {
		_ = C.call_LGBM_BoosterFree(
			C.BoosterHandle(handle),
		)
	}
}

func boosterAddValidData(handle unsafe.Pointer, datasetHandle unsafe.Pointer) error {
	if handle == nil || datasetHandle == nil {
		return errInvalidHandle
	}

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Add validation data
	ret := C.call_LGBM_BoosterAddValidData(
		C.BoosterHandle(handle),
		C.DatasetHandle(datasetHandle),
	)
	if ret != 0 {
		return getLastError()
	}

	// Done
	return nil
}

func boosterUpdateOneIter(handle unsafe.Pointer) (bool, error) {
	var isFinished int32

	if handle == nil {
		return false, errInvalidHandle
	}

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Update one iteration
	ret := C.call_LGBM_BoosterUpdateOneIter(
		C.BoosterHandle(handle),
		(*C.int)(&isFinished),
	)
	if ret != 0 {
		return false, getLastError()
	}

	// Done
	return isFinished != 0, nil
}

func boosterGetEval(handle unsafe.Pointer, dataIndex int) ([]float64, error) {
	var outLen int32

	if handle == nil {
		return nil, errInvalidHandle
	}

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Get count
	ret := C.call_LGBM_BoosterGetEvalCounts(
		C.BoosterHandle(handle),
		(*C.int)(&outLen),
	)
	if ret != 0 {
		return nil, getLastError()
	}
	if outLen <= 0 {
		return make([]float64, 0), nil
	}

	// Make room for results
	results := make([]float64, int(outLen))

	// Get eval
	ret = C.call_LGBM_BoosterGetEval(
		C.BoosterHandle(handle),
		C.int(dataIndex),
		(*C.int)(&outLen),
		(*C.double)(unsafe.Pointer(&results[0])),
	)
	if ret != 0 {
		return nil, getLastError()
	}

	// Done
	return results, nil
}

func boosterSaveModelToString(handle unsafe.Pointer, featureImportance int) (string, error) {
	var outLen int64

	if handle == nil {
		return "", errInvalidHandle
	}
	if featureImportance != C.C_API_FEATURE_IMPORTANCE_SPLIT && featureImportance != C.C_API_FEATURE_IMPORTANCE_GAIN {
		return "", errors.New("invalid feature importance parameter")
	}

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Create room for output
	buf := make([]byte, 32768)
	cBuf := (*C.char)(unsafe.Pointer(&buf[0]))

	// Save the model into a string
	ret := C.call_LGBM_BoosterSaveModelToString(
		C.BoosterHandle(handle),
		C.int(0),
		C.int(0),
		C.int(featureImportance),
		C.longlong(len(buf)),
		(*C.longlong)(&outLen),
		cBuf,
	)
	// If not enough space
	if ret == 0 && int(outLen) >= len(buf) {
		// Build a new room with sufficient space
		buf = make([]byte, int(outLen)+1)
		cBuf = (*C.char)(unsafe.Pointer(&buf[0]))

		// Save the model into a string
		ret = C.call_LGBM_BoosterSaveModelToString(
			C.BoosterHandle(handle),
			C.int(0),
			C.int(0),
			C.int(featureImportance),
			C.longlong(len(buf)),
			(*C.longlong)(&outLen),
			cBuf,
		)
	}
	if ret != 0 {
		return "", getLastError()
	}

	// Done
	return string(buf[:int(outLen)]), nil
}

func boosterLoadModelFromString(data string) (unsafe.Pointer, error) {
	var outNumIterations int32
	var handle unsafe.Pointer

	// Initialize engine
	if err := lazyInitialize(); err != nil {
		return nil, err
	}

	// Convert parameters
	cData := C.CString(data)
	defer C.free(unsafe.Pointer(cData))

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Create the model from the provided data
	ret := C.call_LGBM_BoosterLoadModelFromString(
		cData,
		(*C.int)(&outNumIterations),
		(*C.BoosterHandle)(&handle),
	)
	if ret != 0 {
		return nil, getLastError()
	}

	// Done
	return handle, nil
}

func boosterGetFeaturesCount(handle unsafe.Pointer) (int, error) {
	var featuresCount int32

	if handle == nil {
		return 0, errInvalidHandle
	}

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Get the number of features
	ret := C.call_LGBM_BoosterGetNumFeature(
		C.BoosterHandle(handle),
		(*C.int)(&featuresCount),
	)
	if ret != 0 {
		return 0, getLastError()
	}

	// Done
	return int(featuresCount), nil
}

func boosterGetClassesCount(handle unsafe.Pointer) (int, error) {
	var classesCount int32

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Get the number of classes
	ret := C.call_LGBM_BoosterGetNumClasses(
		C.BoosterHandle(handle),
		(*C.int)(&classesCount),
	)
	if ret != 0 {
		return 0, getLastError()
	}

	// Done
	return int(classesCount), nil
}

func boosterPredictForMatSingleRowFastInit(handle unsafe.Pointer, rawScore bool, parameters string) (unsafe.Pointer, error) {
	var featuresCount int32
	var fastPredictPtr unsafe.Pointer

	if handle == nil {
		return nil, errInvalidHandle
	}

	// Convert parameters
	cParams := C.CString(parameters)
	defer C.free(unsafe.Pointer(cParams))

	predictType := C.C_API_PREDICT_NORMAL
	if rawScore {
		predictType = C.C_API_PREDICT_RAW_SCORE
	}

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Get the number of expected features
	ret := C.call_LGBM_BoosterGetNumFeature(
		C.BoosterHandle(handle),
		(*C.int)(&featuresCount),
	)
	if ret != 0 {
		return nil, getLastError()
	}

	// Create the predictor object
	ret = C.call_LGBM_BoosterPredictForMatSingleRowFastInit(
		C.BoosterHandle(handle),
		C.int(predictType),
		C.int(0),
		C.int(-1),
		C.int(C.C_API_DTYPE_FLOAT64),
		C.int(featuresCount),
		cParams,
		(*C.FastConfigHandle)(&fastPredictPtr),
	)
	if ret != 0 {
		return nil, getLastError()
	}

	// Done
	return fastPredictPtr, nil
}

func boosterPredictForMatSingleRowFast(handle unsafe.Pointer, data []float64, results []float64) error {
	var outLen int64

	if handle == nil {
		return errInvalidHandle
	}

	// Lock thread
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Do prediction
	ret := C.call_LGBM_BoosterPredictForMatSingleRowFast(
		C.FastConfigHandle(handle),
		unsafe.Pointer(&data[0]),
		(*C.longlong)(&outLen),
		(*C.double)(unsafe.Pointer(&results[0])),
	)
	runtime.KeepAlive(data) // Yes, keep-alive should be placed after the position where is used
	if ret != 0 {
		return getLastError()
	}

	// Done
	return nil
}

func predictFastConfigFree(handle unsafe.Pointer) {
	if handle != nil {
		C.call_LGBM_FastConfigFree(C.FastConfigHandle(handle))
	}
}

func getLastError() error {
	msg := C.call_LGBM_GetLastError()
	if msg == nil {
		return nil
	}
	return fmt.Errorf("LightGBM error: %v", C.GoString(msg))
}

func savePointers(
	ptr_LGBM_GetLastError unsafe.Pointer,
	ptr_LGBM_RegisterLogCallback unsafe.Pointer,
	ptr_LGBM_DatasetCreateFromMat unsafe.Pointer,
	ptr_LGBM_DatasetFree unsafe.Pointer,
	ptr_LGBM_DatasetSetField unsafe.Pointer,
	ptr_LGBM_BoosterCreate unsafe.Pointer,
	ptr_LGBM_BoosterFree unsafe.Pointer,
	ptr_LGBM_BoosterAddValidData unsafe.Pointer,
	ptr_LGBM_BoosterUpdateOneIter unsafe.Pointer,
	ptr_LGBM_BoosterGetEval unsafe.Pointer,
	ptr_LGBM_BoosterGetEvalCounts unsafe.Pointer,
	ptr_LGBM_BoosterGetNumFeature unsafe.Pointer,
	ptr_LGBM_BoosterGetNumClasses unsafe.Pointer,
	ptr_LGBM_BoosterGetNumPredict unsafe.Pointer,
	ptr_LGBM_BoosterSaveModelToString unsafe.Pointer,
	ptr_LGBM_BoosterLoadModelFromString unsafe.Pointer,
	ptr_LGBM_BoosterPredictForMatSingleRowFastInit unsafe.Pointer,
	ptr_LGBM_BoosterPredictForMatSingleRowFast unsafe.Pointer,
	ptr_LGBM_FastConfigFree unsafe.Pointer,
) {
	C.savePointers(
		ptr_LGBM_GetLastError,
		ptr_LGBM_RegisterLogCallback,
		ptr_LGBM_DatasetCreateFromMat,
		ptr_LGBM_DatasetFree,
		ptr_LGBM_DatasetSetField,
		ptr_LGBM_BoosterCreate,
		ptr_LGBM_BoosterFree,
		ptr_LGBM_BoosterAddValidData,
		ptr_LGBM_BoosterUpdateOneIter,
		ptr_LGBM_BoosterGetEval,
		ptr_LGBM_BoosterGetEvalCounts,
		ptr_LGBM_BoosterGetNumFeature,
		ptr_LGBM_BoosterGetNumClasses,
		ptr_LGBM_BoosterGetNumPredict,
		ptr_LGBM_BoosterSaveModelToString,
		ptr_LGBM_BoosterLoadModelFromString,
		ptr_LGBM_BoosterPredictForMatSingleRowFastInit,
		ptr_LGBM_BoosterPredictForMatSingleRowFast,
		ptr_LGBM_FastConfigFree,
	)
}
