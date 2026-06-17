package com.qinhan.controller;


import com.qinhan.model.ClusterStatus;
import com.qinhan.service.MemberClusterService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 管理与展示各个成员集群的状态信息
 */
@Slf4j
@RestController
@RequestMapping("/api/clusters")
public class MemberClusterController {

    @Autowired
    private MemberClusterService memberClusterService;

    /**
     * 接收上报的数据
     *
     * @param status 由成员集群进行上报的数据
     * @return 数据来自哪一个集群
     */
    @PostMapping("/report")
    public String reportStatus(@RequestBody ClusterStatus status) {
        log.debug("接收上报的原始数据:{}", status);
        memberClusterService.updateClusterStatus(status);
        return "✅ Received status from cluster: " + status.getClusterName();
    }

    // 获取全部集群状态
    @GetMapping("/all")
    public List<ClusterStatus> getAllStatus() {
        log.info("查看所有集群状态");
        return memberClusterService.getAllClusterStatus();
    }
}
