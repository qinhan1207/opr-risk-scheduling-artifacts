package com.qinhan.service;

import com.qinhan.model.ClusterStatus;

/**
 * 聚合 LSA 上报的原始网络探测数据。
 */
public interface NetworkAggregationService {

    /**
     * 将原始探测结果聚合为 GS 统一使用的网络指标。
     */
    ClusterStatus aggregate(ClusterStatus status);
}
