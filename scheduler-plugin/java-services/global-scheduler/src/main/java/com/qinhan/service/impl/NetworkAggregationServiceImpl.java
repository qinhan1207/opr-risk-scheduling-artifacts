package com.qinhan.service.impl;

import com.qinhan.model.ClusterStatus;
import com.qinhan.model.RawNetworkStats;
import com.qinhan.service.NetworkAggregationService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@Service
public class NetworkAggregationServiceImpl implements NetworkAggregationService {

    private static final double MAX_PENALTY_LATENCY = 2000.0;

    @Override
    public ClusterStatus aggregate(ClusterStatus status) {
        Map<String, RawNetworkStats> rawMap = status.getPeerRawStats();

        if (rawMap == null || rawMap.isEmpty()) {
            status.setNetworkLatency(MAX_PENALTY_LATENCY);
            status.setPacketLossRate(100.0);
            status.setPeerLatencyMap(new HashMap<>());
            return status;
        }

        Map<String, Double> peerLatencyMap = new HashMap<>();
        double minLatency = Double.MAX_VALUE;
        boolean foundValidPath = false;
        double totalRawLossRate = 0.0;
        int countedLinks = 0;

        for (Map.Entry<String, RawNetworkStats> entry : rawMap.entrySet()) {
            RawNetworkStats stats = entry.getValue();
            if (stats == null) {
                continue;
            }

            double avgLatency = stats.getAvgLatency();
            double lossRate = stats.getLossRate();

            if (avgLatency > 0) {
                minLatency = Math.min(minLatency, avgLatency);
                foundValidPath = true;
            }

            totalRawLossRate += lossRate;
            countedLinks++;
            peerLatencyMap.put(entry.getKey(), avgLatency);
        }

        double finalLatency = foundValidPath ? minLatency : MAX_PENALTY_LATENCY;
        double finalLossRate = countedLinks > 0 ? (totalRawLossRate / countedLinks) : 100.0;

        status.setNetworkLatency(finalLatency);
        status.setPacketLossRate(finalLossRate);
        status.setPeerLatencyMap(peerLatencyMap);

        if (foundValidPath) {
            log.debug("[聚合策略] 集群={} 原始={} 选取Min={}",
                    status.getClusterName(), peerLatencyMap, finalLatency);
        }

        return status;
    }
}
