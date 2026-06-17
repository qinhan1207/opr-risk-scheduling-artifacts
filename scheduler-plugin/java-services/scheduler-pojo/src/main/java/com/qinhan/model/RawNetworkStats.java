package com.qinhan.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 单次网络探测的原始结果，供 GS 聚合使用。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RawNetworkStats {
    /** 平均 RTT，单位毫秒 */
    private double avgLatency;

    /** 丢包率，百分比 0-100 */
    private double lossRate;
}
