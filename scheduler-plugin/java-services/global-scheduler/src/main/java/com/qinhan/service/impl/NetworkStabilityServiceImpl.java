package com.qinhan.service.impl;

import com.qinhan.model.ClusterStatus;
import com.qinhan.model.PredictionResult;
import com.qinhan.service.NetworkStabilityService;
import com.qinhan.util.OprRiskPredictor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class NetworkStabilityServiceImpl implements NetworkStabilityService {

    @Value("${experiment.risk-alpha:0.5}")
    private double riskAlpha;

    private static final double MAX_LATENCY_NORM = 300.0;

    @Override
    public void evaluateStability(ClusterStatus status) {
        String clusterName = status.getClusterName();
        double currentLatency = status.getNetworkLatency();
        List<Double> latencyWindow = status.getLatencyWindow();

        log.info("GS received observations: cluster={} latency={}ms loss={} latencyWindow={}",
                clusterName,
                String.format("%.2f", currentLatency),
                String.format("%.2f", status.getPacketLossRate()),
                latencyWindow);

        PredictionResult prediction;
        String inferenceMode;
        if (latencyWindow != null && !latencyWindow.isEmpty()) {
            inferenceMode = "latency-window";
            prediction = OprRiskPredictor.predictByLatencyWindow(clusterName, latencyWindow);
            if (!prediction.isSuccess()) {
                log.warn("Window inference failed, fallback to single-point inference: cluster={}, msg={}",
                        clusterName, prediction.getMessage());
                inferenceMode = "single-fallback";
                prediction = OprRiskPredictor.predict(clusterName, currentLatency);
            }
        } else {
            log.warn("latencyWindow is empty, fallback to single-point inference: cluster={}", clusterName);
            inferenceMode = "single-no-window";
            prediction = OprRiskPredictor.predict(clusterName, currentLatency);
        }

        double riskProb = prediction.getProbability();
        int windowSize = latencyWindow == null ? 0 : latencyWindow.size();
        log.info("Python OPR inference: cluster={} prob={} fault={} mode={} windowSize={} success={} msg={}",
                clusterName,
                String.format("%.4f", riskProb),
                prediction.isFault(),
                inferenceMode,
                windowSize,
                prediction.isSuccess(),
                prediction.getMessage());

        double networkCost = Math.min(1.0, currentLatency / MAX_LATENCY_NORM);
        double effectiveRisk = riskProb > 0.8 ? 1.0 : riskProb;
        double fusedCost = (1 - riskAlpha) * networkCost + riskAlpha * effectiveRisk;
        double score = (1.0 - Math.min(1.0, fusedCost)) * 100.0;

        status.setStabilityScore(score);
        status.setRemark(String.format(
                "Latency:%.2fms Risk:%.4f Cost:%.3f -> Score:%.1f",
                currentLatency,
                riskProb,
                fusedCost,
                score));
    }
}
