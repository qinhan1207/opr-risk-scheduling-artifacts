package com.qinhan.service;

import com.qinhan.model.ClusterStatus;

/**
 * Evaluates member-cluster stability by calling the Python OPR risk predictor
 * and converting the returned probability into a scheduler-side score.
 */
public interface NetworkStabilityService {

    /**
     * Updates the given status with the latest stability score and explanation.
     *
     * @param status aggregated member-cluster observations
     */
    void evaluateStability(ClusterStatus status);
}
