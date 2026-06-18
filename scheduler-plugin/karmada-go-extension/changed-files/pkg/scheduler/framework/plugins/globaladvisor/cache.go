package globaladvisor

import (
	"sync"
	"time"
)

type cacheEntry struct {
	score     float64
	timestamp time.Time
}

type simpleCache struct {
	store map[string]cacheEntry
	mu    sync.RWMutex
	ttl   time.Duration
}

func newSimpleCache(ttl time.Duration) *simpleCache {
	return &simpleCache{
		store: make(map[string]cacheEntry),
		ttl:   ttl,
	}
}

func (c *simpleCache) Get(key string) (float64, bool) {
	c.mu.RLock()
	e, ok := c.store[key]
	c.mu.RUnlock()
	if !ok {
		return 0, false
	}
	if time.Since(e.timestamp) > c.ttl {
		c.mu.Lock()
		delete(c.store, key)
		c.mu.Unlock()
		return 0, false
	}
	return e.score, true
}

func (c *simpleCache) Set(key string, score float64) {
	c.mu.Lock()
	c.store[key] = cacheEntry{score: score, timestamp: time.Now()}
	c.mu.Unlock()
}
