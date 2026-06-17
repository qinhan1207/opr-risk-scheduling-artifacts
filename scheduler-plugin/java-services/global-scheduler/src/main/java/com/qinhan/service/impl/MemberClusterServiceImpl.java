package com.qinhan.service.impl;

import com.qinhan.model.ClusterStatus;
import com.qinhan.service.MemberClusterService;
import com.qinhan.service.NetworkAggregationService;
import com.qinhan.service.NetworkStabilityService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Service
@RequiredArgsConstructor
public class MemberClusterServiceImpl implements MemberClusterService {

    private final ConcurrentHashMap<String, ClusterStatus> clusterMap = new ConcurrentHashMap<>();

    private final NetworkStabilityService networkStabilityService;
    private final NetworkAggregationService networkAggregationService;

    @Override
    public void updateClusterStatus(ClusterStatus status) {
        if (needCloudAggregation(status)) {
            status = networkAggregationService.aggregate(status);
            log.debug(
                    "Cloud-side aggregation fallback: cluster={} latency={}ms loss={} peerLatency={}",
                    status.getClusterName(),
                    String.format("%.2f", status.getNetworkLatency()),
                    String.format("%.2f", status.getPacketLossRate()),
                    status.getPeerLatencyMap()
            );
        } else {
            log.debug(
                    "Using edge-aggregated observations: cluster={} latency={}ms loss={} peerRawStats={}",
                    status.getClusterName(),
                    String.format("%.2f", status.getNetworkLatency()),
                    String.format("%.2f", status.getPacketLossRate()),
                    status.getPeerRawStats() == null ? 0 : status.getPeerRawStats().size()
            );
        }

        networkStabilityService.evaluateStability(status);
        clusterMap.put(status.getClusterName(), status);

        log.debug(
                "Cluster status updated: cluster={} stabilityScore={}",
                status.getClusterName(),
                String.format("%.0f", status.getStabilityScore())
        );
    }

    @Override
    public List<ClusterStatus> getAllClusterStatus() {
        return new ArrayList<>(clusterMap.values());
    }

    private boolean needCloudAggregation(ClusterStatus status) {
        return status.getNetworkLatency() <= 0 || status.getPacketLossRate() < 0;
    }
}
