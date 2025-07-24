package lightgbm

import (
	"regexp"
	"strings"
	"sync"
)

// -----------------------------------------------------------------------------

type LoggerCallback func(msgType string, msg string)

// -----------------------------------------------------------------------------

var loggerMtx sync.RWMutex
var loggerCB LoggerCallback
var prefixRegex = regexp.MustCompile(`^\[LightGBM\]\s*\[([^\]]+)\]$`)

// -----------------------------------------------------------------------------

func LoggerSetCallback(cb LoggerCallback) {
	loggerMtx.Lock()
	defer loggerMtx.Unlock()

	loggerCB = cb
}

func loggerCollector() {
	var msgType string

	fullMsg := make([]string, 0)

	for {
		msg := <-loggerCh

		msg = strings.TrimSpace(msg)
		if len(msg) > 0 {
			if match := prefixRegex.FindStringSubmatch(msg); match != nil {
				msgType = strings.ToUpper(match[1])
			} else if len(msgType) > 0 {
				fullMsg = append(fullMsg, msg)
			}
		} else {
			if len(msgType) > 0 && len(fullMsg) > 0 {
				loggerMtx.RLock()
				if loggerCB != nil {
					loggerCB(msgType, strings.Join(fullMsg, " "))
				}
				loggerMtx.RUnlock()
			}
			msgType = ""
			fullMsg = make([]string, 0)
		}
	}
}
