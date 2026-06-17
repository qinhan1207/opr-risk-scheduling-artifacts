package com.qinhan.controller;

import com.qinhan.model.ClusterScore;
import com.qinhan.service.ClusterScoreService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * HTTP endpoints consumed by the Karmada-side scheduler extension.
 */
@Slf4j
@RestController
@RequestMapping("/api/advisor")
@RequiredArgsConstructor
public class ClusterScoreController {

    private final ClusterScoreService clusterScoreService;

    /**
     * Returns the OPR-TSMixer risk-aware score for a candidate member cluster.
     */
    @GetMapping("/score")
    public ClusterScore getClusterScore(
            @RequestParam("cluster") String clusterName,
            @RequestParam(value = "target", required = false) String targetCluster
    ) {
        log.info("Risk-aware scoring request: cluster={}, target={}", clusterName, targetCluster);
        ClusterScore score = clusterScoreService.calculateScore(clusterName, targetCluster);
        log.info("Risk-aware scoring response: {}", score);
        return score;
    }

    /**
     * Returns the network-only score used for comparison experiments.
     */
    @GetMapping("/network-score")
    public ClusterScore getNetworkOnlyScore(
            @RequestParam("cluster") String clusterName,
            @RequestParam(value = "target", required = false) String targetCluster
    ) {
        log.info("Network-only scoring request: cluster={}, target={}", clusterName, targetCluster);
        ClusterScore result = clusterScoreService.calculateNetworkOnlyScore(clusterName, targetCluster);
        log.info("Network-only scoring response: {}", result);
        return result;
    }

    @GetMapping("/ping")
    public String ping() {
        return "Global Scheduler is running";
    }
}
