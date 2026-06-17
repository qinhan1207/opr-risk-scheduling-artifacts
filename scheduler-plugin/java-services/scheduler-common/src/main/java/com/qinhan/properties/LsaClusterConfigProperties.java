package com.qinhan.properties;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.List;

/**
 * LSA多集群配置
 * 从 application.yml 读取 LSA 多集群配置信息
 */
@Data
@ConfigurationProperties(prefix = "lsa")
public class LsaClusterConfigProperties {

    /**
     * 运行模式:
     * - standalone: (默认) 集中式模式，读取 configs 列表连接多个集群
     * - distributed: 分布式模式，运行在 Pod 内部，只采集当前集群
     */
    private String mode = "standalone";

    /**
     * 当前集群名称 (仅在 distributed 模式下生效)
     * 通常通过环境变量注入
     */
    private String currentClusterName;

    /**
     * 采样周期（毫秒）：高频采样并更新 latency window
     */
    private long sampleIntervalMs = 1000;

    /**
     * 上报周期（毫秒）：低频上报窗口与最近状态
     */
    private long reportIntervalMs = 5000;

    /**
     * 窗口大小：上报时携带最近 N 个 member-cluster-level latency 观测值
     */
    private int latencyWindowSize = 10;

    /**
     * 集中式模式下的集群配置列表
     */
    private Clusters clusters;

    @Data
    public static class Clusters {
        private List<ClusterConfig> configs;
    }

    @Data
    public static class ClusterConfig {
        private String name;            // 集群名称
        private String kubeconfigPath;  // kubeconfig 路径
    }
}
