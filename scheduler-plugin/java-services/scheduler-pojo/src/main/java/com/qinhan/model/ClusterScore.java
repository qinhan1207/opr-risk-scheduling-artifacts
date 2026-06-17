package com.qinhan.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Karmada 调度插件可直接使用的简化结果模型
 * 表示 GS 对集群的综合评分
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ClusterScore {
    /** 集群名称 */
    private String clusterName;

    /**
     * 兼容旧字段：综合得分（0~100）。
     * 建议优先读取 finalScore。
     */
    private double healthScore;

    /** 最终调度分（0~100） */
    private double finalScore;

    /** 评分说明 */
    private String reason;
}