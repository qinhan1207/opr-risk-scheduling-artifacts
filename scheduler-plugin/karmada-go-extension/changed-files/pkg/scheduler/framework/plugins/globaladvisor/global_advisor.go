package globaladvisor

import (
	"context"
	"fmt"
	"os"
	"time"

	clusterv1alpha1 "github.com/karmada-io/karmada/pkg/apis/cluster/v1alpha1"
	workv1alpha2 "github.com/karmada-io/karmada/pkg/apis/work/v1alpha2"
	"github.com/karmada-io/karmada/pkg/scheduler/framework"
	"k8s.io/klog/v2"
)

const (
	Name           = "GlobalAdvisor"
	defaultGSURL   = "http://127.0.0.1:9090"
	defaultTimeout = 300 * time.Millisecond
	defaultRetry   = 1
	defaultBackoff = 100 * time.Millisecond
	defaultTTL     = 3 * time.Second
	defaultScore   = 50.0

	AffinityLabelKey = "scheduler.qinhan.io/affinity-target"

	// ModeEnvKey controls which extension points are exposed by this plugin.
	// Supported values:
	//   - both: expose Filter + Score
	//   - filter-only: expose Filter only
	//   - score-only: expose Score only
	ModeEnvKey = "GLOBAL_ADVISOR_MODE"
	modeBoth   = "both"
	modeFilter = "filter-only"
	modeScore  = "score-only"
)

// GlobalAdvisor calls the Java global scheduler service and converts the
// returned risk-aware score into Karmada framework decisions.
type GlobalAdvisor struct {
	scoreClient  *ScoreClient
	cache        *simpleCache
	defaultScore float64
}

var _ framework.FilterPlugin = &GlobalAdvisor{}
var _ framework.ScorePlugin = &GlobalAdvisor{}
var _ framework.FilterPlugin = &globalAdvisorFilterOnly{}
var _ framework.ScorePlugin = &globalAdvisorScoreOnly{}

type globalAdvisorFilterOnly struct {
	core *GlobalAdvisor
}

func (p *globalAdvisorFilterOnly) Name() string {
	return Name
}

func (p *globalAdvisorFilterOnly) Filter(
	ctx context.Context,
	bindingSpec *workv1alpha2.ResourceBindingSpec,
	bindingStatus *workv1alpha2.ResourceBindingStatus,
	cluster *clusterv1alpha1.Cluster,
) *framework.Result {
	return p.core.Filter(ctx, bindingSpec, bindingStatus, cluster)
}

type globalAdvisorScoreOnly struct {
	core *GlobalAdvisor
}

func (p *globalAdvisorScoreOnly) Name() string {
	return Name
}

func (p *globalAdvisorScoreOnly) Score(
	ctx context.Context,
	spec *workv1alpha2.ResourceBindingSpec,
	cluster *clusterv1alpha1.Cluster,
) (int64, *framework.Result) {
	return p.core.Score(ctx, spec, cluster)
}

func (p *globalAdvisorScoreOnly) ScoreExtensions() framework.ScoreExtensions {
	return p
}

func (p *globalAdvisorScoreOnly) NormalizeScore(_ context.Context, _ framework.ClusterScoreList) *framework.Result {
	return framework.NewResult(framework.Success)
}

// New creates a GlobalAdvisor plugin instance.
func New() (framework.Plugin, error) {
	gsURL := os.Getenv("GLOBAL_SCHEDULER_URL")
	if gsURL == "" {
		gsURL = defaultGSURL
	}
	klog.Infof("[GlobalAdvisor] Connecting to Global Scheduler at: %s", gsURL)

	client := NewScoreClient(gsURL, defaultTimeout, defaultRetry, defaultBackoff)
	cache := newSimpleCache(defaultTTL)

	core := &GlobalAdvisor{
		scoreClient:  client,
		cache:        cache,
		defaultScore: defaultScore,
	}

	mode := os.Getenv(ModeEnvKey)
	if mode == "" {
		mode = modeBoth
	}
	switch mode {
	case modeFilter:
		klog.Infof("[GlobalAdvisor] mode=%s, expose Filter only", modeFilter)
		return &globalAdvisorFilterOnly{core: core}, nil
	case modeScore:
		klog.Infof("[GlobalAdvisor] mode=%s, expose Score only", modeScore)
		return &globalAdvisorScoreOnly{core: core}, nil
	default:
		if mode != modeBoth {
			klog.Warningf("[GlobalAdvisor] unknown %s=%q, fallback to %s", ModeEnvKey, mode, modeBoth)
		}
		return core, nil
	}
}

func (g *GlobalAdvisor) Name() string {
	return Name
}

// Filter suppresses candidates whose score from the Java global scheduler is
// below the configured low-score cutoff. External service failures are treated
// as soft failures so that Karmada remains schedulable.
func (g *GlobalAdvisor) Filter(
	ctx context.Context,
	bindingSpec *workv1alpha2.ResourceBindingSpec,
	_ *workv1alpha2.ResourceBindingStatus,
	cluster *clusterv1alpha1.Cluster,
) *framework.Result {
	clusterName := cluster.Name
	targetCluster := detectTargetCluster(bindingSpec)
	if targetCluster == "" {
		if t := os.Getenv("TEST_AFFINITY_TARGET"); t != "" {
			targetCluster = t
		}
	}

	klog.V(3).Infof("[GlobalAdvisor] Filter called for cluster=%s, target=%s", clusterName, targetCluster)

	ctxTimeout, cancel := context.WithTimeout(ctx, defaultTimeout)
	defer cancel()

	scoreResp, err := g.scoreClient.GetScore(ctxTimeout, clusterName, targetCluster)
	if err != nil {
		klog.Warningf("[GlobalAdvisor] Filter: failed to get score for cluster=%s: %v; allow scheduling", clusterName, err)
		return framework.NewResult(framework.Success)
	}

	score := clampScore(scoreResp.HealthScore)
	if scoreResp.Reason == "cluster status data not found" {
		klog.Warningf("[GlobalAdvisor] Filter: no cluster status data for cluster=%s, allow scheduling", clusterName)
		return framework.NewResult(framework.Success)
	}

	if score <= 55 {
		klog.Infof("[GlobalAdvisor] Filter: suppress cluster=%s score=%.2f reason=%s", clusterName, score, scoreResp.Reason)
		return framework.NewResult(
			framework.Unschedulable,
			fmt.Sprintf("GlobalAdvisor suppressed cluster %s due to low health score %.2f, reason: %s", clusterName, score, scoreResp.Reason),
		)
	}

	return framework.NewResult(framework.Success)
}

// Score returns the Java global scheduler score for a candidate cluster.
func (g *GlobalAdvisor) Score(ctx context.Context, spec *workv1alpha2.ResourceBindingSpec, cluster *clusterv1alpha1.Cluster) (int64, *framework.Result) {
	clusterName := cluster.Name
	targetCluster := detectTargetCluster(spec)
	if targetCluster == "" {
		if t := os.Getenv("TEST_AFFINITY_TARGET"); t != "" {
			targetCluster = t
		}
	}

	klog.V(3).Infof("[GlobalAdvisor] Score called for cluster=%s, target=%s", clusterName, targetCluster)

	if targetCluster == "" {
		if s, ok := g.cache.Get(clusterName); ok {
			return int64(s), framework.NewResult(framework.Success)
		}
	}

	ctxTimeout, cancel := context.WithTimeout(ctx, defaultTimeout)
	defer cancel()

	scoreResp, err := g.scoreClient.GetScore(ctxTimeout, clusterName, targetCluster)
	if err != nil {
		klog.Warningf("[GlobalAdvisor] failed to get score for cluster=%s: %v; fallback", clusterName, err)
		return int64(g.defaultScore), framework.NewResult(framework.Success)
	}

	score := clampScore(scoreResp.HealthScore)
	if targetCluster == "" {
		g.cache.Set(clusterName, score)
	}

	klog.Infof("[GlobalAdvisor] got score cluster=%s score=%.2f reason=%s", clusterName, score, scoreResp.Reason)
	return int64(score), framework.NewResult(framework.Success)
}

func (g *GlobalAdvisor) ScoreExtensions() framework.ScoreExtensions {
	return g
}

func (g *GlobalAdvisor) NormalizeScore(_ context.Context, _ framework.ClusterScoreList) *framework.Result {
	return framework.NewResult(framework.Success)
}

func clampScore(score float64) float64 {
	if score < 0 {
		return 0
	}
	if score > 100 {
		return 100
	}
	return score
}

func detectTargetCluster(spec *workv1alpha2.ResourceBindingSpec) string {
	if spec == nil {
		return ""
	}

	if target := targetFromReplicaRequirements(spec.ReplicaRequirements); target != "" {
		return target
	}

	for _, comp := range spec.Components {
		if target := targetFromComponentRequirements(comp.ReplicaRequirements); target != "" {
			return target
		}
	}

	return ""
}

func targetFromReplicaRequirements(req *workv1alpha2.ReplicaRequirements) string {
	if req == nil {
		return ""
	}
	return targetFromNodeClaim(req.NodeClaim)
}

func targetFromComponentRequirements(req *workv1alpha2.ComponentReplicaRequirements) string {
	if req == nil {
		return ""
	}
	return targetFromNodeClaim(req.NodeClaim)
}

func targetFromNodeClaim(claim *workv1alpha2.NodeClaim) string {
	if claim == nil || claim.NodeSelector == nil {
		return ""
	}
	if target, ok := claim.NodeSelector[AffinityLabelKey]; ok {
		return target
	}
	return ""
}
