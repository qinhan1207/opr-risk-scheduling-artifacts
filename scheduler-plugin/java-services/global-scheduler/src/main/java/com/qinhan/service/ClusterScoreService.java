package com.qinhan.service;

import com.qinhan.model.ClusterScore;

/**
 * Computes scheduler-consumable scores for Karmada candidate clusters.
 */
public interface ClusterScoreService {

    /**
     * Computes the risk-aware score backed by the Python OPR-TSMixer predictor.
     *
     * @param clusterName candidate member-cluster name
     * @param targetCluster optional target cluster context
     * @return score and scoring explanation
     */
    ClusterScore calculateScore(String clusterName, String targetCluster);

    /**
     * Computes a network-only score for comparison and ablation experiments.
     *
     * @param clusterName candidate member-cluster name
     * @param targetCluster optional target cluster context
     * @return network-only score and scoring explanation
     */
    ClusterScore calculateNetworkOnlyScore(String clusterName, String targetCluster);
}
