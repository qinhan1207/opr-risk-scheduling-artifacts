package com.qinhan.service.impl;

import com.qinhan.model.ClusterStatus;
import com.qinhan.model.RawNetworkStats;
import com.qinhan.properties.LsaClusterConfigProperties;
import com.qinhan.service.ClusterMonitorService;
import com.qinhan.util.K8sClientUtil;
import com.qinhan.util.NetworkUtils;
import io.kubernetes.client.openapi.ApiClient;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Service
public class ClusterMonitorServiceImpl implements ClusterMonitorService {

    @Resource
    private LsaClusterConfigProperties lsaProperties;

    private final K8sClientUtil k8sClientUtil;

    public ClusterMonitorServiceImpl(K8sClientUtil k8sClientUtil) {
        this.k8sClientUtil = k8sClientUtil;
    }

    @Override
    public void testClusterConnection(String kubeconfigPath) {
        try {
            ApiClient client = k8sClientUtil.getClient(kubeconfigPath);
            log.info("Connected to cluster [{}]", kubeconfigPath);
        } catch (Exception e) {
            log.error("Cannot connect to cluster [{}]", kubeconfigPath, e);
        }
    }

    @Override
    public ClusterStatus collectClusterStatus(String kubeconfigPath) {
        try {
            ApiClient client = k8sClientUtil.getClient(kubeconfigPath);

            Map<String, NetworkUtils.NetworkStats> rawStatsMap = probeNeighborsStats();
            Map<String, RawNetworkStats> peerRawStats = new HashMap<>();
            rawStatsMap.forEach((name, stats) -> peerRawStats.put(name,
                    RawNetworkStats.builder()
                            .avgLatency(stats.getAvgLatency())
                            .lossRate(stats.getLossRate())
                            .build()));

            double networkLatency = aggregateNetworkLatency(peerRawStats);
            double packetLossRate = aggregatePacketLoss(peerRawStats);

            String clusterName = lsaProperties.getCurrentClusterName();
            log.info("[ProbeSummary] cluster={} latency={}ms loss={} peerRawStats={}",
                    clusterName,
                    String.format("%.2f", networkLatency),
                    String.format("%.2f", packetLossRate),
                    peerRawStats);

            return ClusterStatus.builder()
                    .timestamp(Instant.now().toEpochMilli())
                    .peerRawStats(peerRawStats)
                    .networkLatency(networkLatency)
                    .packetLossRate(packetLossRate)
                    .build();

        } catch (Exception e) {
            log.error("Collect cluster status failed: {}", e.getMessage());
            return null;
        }
    }

    private Map<String, String> getNeighborIps() {
        Map<String, String> neighbors = new HashMap<>();
        neighbors.put("member1", "member1-control-plane");
        neighbors.put("member2", "member2-control-plane");
        neighbors.put("member3", "member3-control-plane");
        neighbors.put("member4", "member4-control-plane");
        neighbors.put("member5", "member5-control-plane");

        String myName = lsaProperties.getCurrentClusterName();
        if (myName != null && neighbors.containsKey(myName)) {
            neighbors.remove(myName);
        }
        return neighbors;
    }

    private double aggregateNetworkLatency(Map<String, RawNetworkStats> rawMap) {
        final double penaltyLatency = 2000.0;
        final double validMaxLatency = 5000.0;

        if (rawMap == null || rawMap.isEmpty()) {
            return penaltyLatency;
        }

        double minLatency = Double.MAX_VALUE;
        boolean hasValid = false;

        for (RawNetworkStats stats : rawMap.values()) {
            if (stats == null) {
                continue;
            }

            double latency = stats.getAvgLatency();
            if (Double.isNaN(latency) || latency <= 0 || latency >= validMaxLatency) {
                continue;
            }

            latency = Math.min(latency, penaltyLatency);
            if (latency < minLatency) {
                minLatency = latency;
                hasValid = true;
            }
        }

        return hasValid ? minLatency : penaltyLatency;
    }

    private double aggregatePacketLoss(Map<String, RawNetworkStats> rawMap) {
        if (rawMap == null || rawMap.isEmpty()) {
            return 100.0;
        }

        double healthyCount = 0;
        for (RawNetworkStats stats : rawMap.values()) {
            if (stats.getLossRate() < 0.1 && stats.getAvgLatency() < 500) {
                healthyCount++;
            }
        }

        if (healthyCount > 0) {
            return 0.0;
        }

        double totalLoss = 0.0;
        for (RawNetworkStats stats : rawMap.values()) {
            totalLoss += stats.getLossRate();
        }
        return rawMap.size() > 0 ? totalLoss / rawMap.size() : 100.0;
    }

    public Map<String, NetworkUtils.NetworkStats> probeNeighborsStats() {
        Map<String, NetworkUtils.NetworkStats> result = new ConcurrentHashMap<>();
        Map<String, String> neighbors = getNeighborIps();

        neighbors.entrySet().parallelStream().forEach(entry -> {
            String name = entry.getKey();
            String target = entry.getValue();
            NetworkUtils.NetworkStats stats = NetworkUtils.ping(target, 5, 2);
            result.put(name, stats);
            if (stats.getLossRate() > 0) {
                log.info("[Probe] cluster={} target={} loss={} latency={}ms",
                        name, target, stats.getLossRate(), stats.getAvgLatency());
            }
        });
        return result;
    }
}
