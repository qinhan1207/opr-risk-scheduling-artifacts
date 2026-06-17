package com.qinhan.client;

import com.qinhan.model.ClusterStatus;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;


/**
 * gs客户端，用于连接gs服务，向gs发送成员集群信息
 */
@Slf4j
@Component
public class GlobalSchedulerClient {

    @Value("${global.scheduler.url:http://localhost:8088}")
    private String globalSchedulerUrl;

    private final RestTemplate restTemplate;

    public GlobalSchedulerClient(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    /**
     * 向 GC 上报集群状态
     */
    public void sendClusterStatus(ClusterStatus status) {
        try {
            String url = globalSchedulerUrl + "/api/clusters/report";
            ResponseEntity<String> response = restTemplate.postForEntity(url, status, String.class);
            log.debug("📤 集群 [{}] 状态上报响应: {}", status.getClusterName(), response.getBody());
        } catch (Exception e) {
            log.error("❌ 上报集群 [{}] 状态失败: {}", status.getClusterName(), e.getMessage());
        }
    }
}
