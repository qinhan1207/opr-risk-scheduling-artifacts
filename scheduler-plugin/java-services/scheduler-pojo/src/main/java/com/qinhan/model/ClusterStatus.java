package com.qinhan.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

/**
 * Status reported by a Local Scheduler Agent for one member cluster.
 *
 * The current OPR path keeps Java responsible for observation collection and
 * window maintenance only. OPR feature construction is performed by the Python
 * inference service so that online inference stays aligned with training code.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ClusterStatus {

    /** Member-cluster name, for example member1 or member3. */
    private String clusterName;

    /** Agent collection timestamp in milliseconds. */
    private long timestamp;

    /** Aggregated member-cluster-level latency observation in milliseconds. */
    private double networkLatency;

    /** Aggregated packet loss rate in percent. */
    private double packetLossRate;

    /** Raw probe results to peer clusters. */
    private Map<String, RawNetworkStats> peerRawStats;

    /** Peer latency map retained for cloud-side aggregation compatibility. */
    private Map<String, Double> peerLatencyMap;

    /** Recent latency observations used by the Python OPR inference service. */
    private List<Double> latencyWindow;

    /** Scheduler-consumable stability score computed by the global service. */
    private double stabilityScore;

    /** Human-readable scoring explanation. */
    private String remark;
}
