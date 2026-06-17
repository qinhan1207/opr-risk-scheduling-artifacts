package com.qinhan.service.impl;

import com.qinhan.client.GlobalSchedulerClient;
import com.qinhan.model.ClusterStatus;
import com.qinhan.properties.LsaClusterConfigProperties;
import com.qinhan.service.ClusterMonitorService;
import com.qinhan.service.ReportService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Service
@RequiredArgsConstructor
public class ReportServiceImpl implements ReportService {

    private final ClusterMonitorService clusterMonitorService;
    private final LsaClusterConfigProperties lsaProperties;
    private final GlobalSchedulerClient globalSchedulerClient;

    private final Map<String, ClusterStatus> latestStatusMap = new ConcurrentHashMap<>();
    private final Map<String, Deque<Double>> latencyWindowMap = new ConcurrentHashMap<>();

    @Scheduled(fixedRateString = "${lsa.sample-interval-ms:1000}")
    public void sampleAllClusters() {
        String mode = lsaProperties.getMode();
        if ("distributed".equalsIgnoreCase(mode)) {
            sampleSelf();
        } else {
            sampleConfiguredClusters();
        }
    }

    @Override
    @Scheduled(fixedRateString = "${lsa.report-interval-ms:5000}")
    public void reportAllClusters() {
        if (latestStatusMap.isEmpty()) {
            log.debug("[ReportTask] no sampled data, skip this round");
            return;
        }

        latestStatusMap.forEach((clusterName, latest) -> {
            try {
                ClusterStatus payload = buildPayload(clusterName, latest);
                globalSchedulerClient.sendClusterStatus(payload);
                int windowSize = payload.getLatencyWindow() == null ? 0 : payload.getLatencyWindow().size();
                log.info("Report success: cluster={} latencyWindow={} latestLatency={}ms loss={}",
                        clusterName,
                        windowSize,
                        String.format("%.2f", payload.getNetworkLatency()),
                        String.format("%.2f", payload.getPacketLossRate()));
            } catch (Exception e) {
                log.error("Report cluster [{}] failed: {}", clusterName, e.getMessage());
            }
        });
    }

    private void sampleSelf() {
        String clusterName = lsaProperties.getCurrentClusterName();
        log.debug("[SampleTask] distributed sampling cluster={}", clusterName);

        try {
            ClusterStatus status = clusterMonitorService.collectClusterStatus("in-cluster");
            if (status != null) {
                status.setClusterName(clusterName);
                rememberSample(clusterName, status);
            }
        } catch (Exception e) {
            log.error("Sample local cluster [{}] failed: {}", clusterName, e.getMessage());
        }
    }

    private void sampleConfiguredClusters() {
        if (lsaProperties.getClusters() == null || lsaProperties.getClusters().getConfigs() == null) {
            log.warn("No cluster list configured");
            return;
        }

        List<LsaClusterConfigProperties.ClusterConfig> configs = lsaProperties.getClusters().getConfigs();
        configs.parallelStream().forEach(config -> {
            try {
                ClusterStatus status = clusterMonitorService.collectClusterStatus(config.getKubeconfigPath());
                if (status != null) {
                    status.setClusterName(config.getName());
                    rememberSample(config.getName(), status);
                }
            } catch (Exception e) {
                log.error("Sample cluster [{}] failed: {}", config.getName(), e.getMessage());
            }
        });
    }

    private void rememberSample(String clusterName, ClusterStatus status) {
        latestStatusMap.put(clusterName, status);

        Deque<Double> queue = latencyWindowMap.computeIfAbsent(clusterName, k -> new ArrayDeque<>());
        synchronized (queue) {
            queue.addLast(status.getNetworkLatency());

            int maxWindow = Math.max(1, lsaProperties.getLatencyWindowSize());
            while (queue.size() > maxWindow) {
                queue.removeFirst();
            }
        }
    }

    private ClusterStatus buildPayload(String clusterName, ClusterStatus latest) {
        List<Double> latencyWindow;
        Deque<Double> queue = latencyWindowMap.get(clusterName);
        if (queue == null) {
            latencyWindow = new ArrayList<>();
        } else {
            synchronized (queue) {
                latencyWindow = new ArrayList<>(queue);
            }
        }

        return ClusterStatus.builder()
                .clusterName(clusterName)
                .timestamp(latest.getTimestamp())
                .networkLatency(latest.getNetworkLatency())
                .packetLossRate(latest.getPacketLossRate())
                .peerRawStats(latest.getPeerRawStats())
                .latencyWindow(latencyWindow)
                .build();
    }
}
