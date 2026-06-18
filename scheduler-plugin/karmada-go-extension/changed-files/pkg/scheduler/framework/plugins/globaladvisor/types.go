package globaladvisor

type ClusterScore struct {
	ClusterName string  `json:"clusterName"`
	HealthScore float64 `json:"healthScore"`
	Reason      string  `json:"reason"`
}
