package com.qinhan.service.impl;

import com.qinhan.model.ClusterScore;
import com.qinhan.model.ClusterStatus;
import com.qinhan.service.ClusterScoreService;
import com.qinhan.service.MemberClusterService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Returns scheduler-consumable scores for member clusters.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ClusterScoreServiceImpl implements ClusterScoreService {

    private static final double MAX_LATENCY_NORM = 100.0;

    private final MemberClusterService memberClusterService;

    @Override
    public ClusterScore calculateScore(String clusterName, String targetCluster) {
        ClusterStatus status = findClusterStatus(clusterName);
        if (status == null) {
            log.warn("Cluster status not found: cluster={}", clusterName);
            return ClusterScore.builder()
                    .clusterName(clusterName)
                    .healthScore(0)
                    .finalScore(0)
                    .reason("cluster status not found")
                    .build();
        }

        double finalScore = status.getStabilityScore();
        String reason = status.getRemark();
        if (reason == null || reason.isEmpty()) {
            reason = String.format("Score:%.2f", finalScore);
        }

        ClusterScore result = ClusterScore.builder()
                .clusterName(clusterName)
                .healthScore(finalScore)
                .finalScore(finalScore)
                .reason(reason)
                .build();

        log.info("[FinalScore] cluster={} finalScore={} reason={}",
                clusterName,
                String.format("%.2f", finalScore),
                reason);

        return result;
    }

    @Override
    public ClusterScore calculateNetworkOnlyScore(String clusterName, String targetCluster) {
        ClusterStatus status = findClusterStatus(clusterName);
        if (status == null) {
            log.warn("Cluster status not found for network-only score: cluster={}", clusterName);
            return ClusterScore.builder()
                    .clusterName(clusterName)
                    .healthScore(0)
                    .finalScore(0)
                    .reason("cluster status not found")
                    .build();
        }

        double networkLatency = status.getNetworkLatency();
        double networkCost = Math.min(1.0, networkLatency / MAX_LATENCY_NORM);
        double networkOnlyScore = (1.0 - networkCost) * 100.0;
        String reason = String.format("NetworkLatency:%.2fms -> Cost:%.2f -> Score:%.1f",
                networkLatency,
                networkCost,
                networkOnlyScore);

        log.info("[NetworkOnlyScore] cluster={} finalScore={} reason={}",
                clusterName,
                String.format("%.2f", networkOnlyScore),
                reason);

        return ClusterScore.builder()
                .clusterName(clusterName)
                .healthScore(networkOnlyScore)
                .finalScore(networkOnlyScore)
                .reason(reason)
                .build();
    }

    private ClusterStatus findClusterStatus(String clusterName) {
        String realName = clusterName;
        if (clusterName.startsWith("cluster")) {
            realName = clusterName.replace("cluster", "member");
            log.debug("[NameMapping] requestCluster={} mappedCluster={}", clusterName, realName);
        }

        String searchName = realName;
        List<ClusterStatus> allStatus = memberClusterService.getAllClusterStatus();
        return allStatus.stream()
                .filter(s -> searchName.equals(s.getClusterName()))
                .findFirst()
                .orElse(null);
    }
}
