package globaladvisor

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"time"
)

// ScoreClient calls the Java global scheduler scoring endpoint.
type ScoreClient struct {
	baseURL    string
	httpClient *http.Client
	retry      int
	backoff    time.Duration
}

// NewScoreClient creates a scoring client with timeout and retry settings.
func NewScoreClient(baseURL string, timeout time.Duration, retry int, backoff time.Duration) *ScoreClient {
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   3 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:        100,
		IdleConnTimeout:     90 * time.Second,
		TLSHandshakeTimeout: 5 * time.Second,
	}
	return &ScoreClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout:   timeout,
			Transport: transport,
		},
		retry:   retry,
		backoff: backoff,
	}
}

// GetScore requests the risk-aware score for one candidate cluster.
func (c *ScoreClient) GetScore(ctx context.Context, clusterName string, targetCluster string) (*ClusterScore, error) {
	endpoint := fmt.Sprintf(
		"%s/api/advisor/score?cluster=%s",
		c.baseURL,
		url.QueryEscape(clusterName),
	)
	if targetCluster != "" {
		endpoint = fmt.Sprintf("%s&target=%s", endpoint, url.QueryEscape(targetCluster))
	}

	var lastErr error
	for attempt := 0; attempt <= c.retry; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return nil, err
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(c.backoff):
				continue
			}
		}

		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("status=%d body=%s", resp.StatusCode, string(body))
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(c.backoff):
				continue
			}
		}

		var sc ClusterScore
		if err := json.Unmarshal(body, &sc); err != nil {
			return nil, err
		}
		return &sc, nil
	}
	return nil, lastErr
}
