package globaladvisor

import (
	"github.com/karmada-io/karmada/pkg/scheduler/framework"
)

// NewPlugin is used by the scheduler registry to create the plugin.
func NewPlugin() (framework.Plugin, error) {
	return New()
}
