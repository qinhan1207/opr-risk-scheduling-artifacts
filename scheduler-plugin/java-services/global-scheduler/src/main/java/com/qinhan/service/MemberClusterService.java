package com.qinhan.service;


import com.qinhan.model.ClusterStatus;

import java.util.List;

public interface MemberClusterService {

    /**
     * 更新集群状态
     */
    void updateClusterStatus(ClusterStatus status);

    /**
     * 查看所有集群
     * @return 所有集群信息
     */
    List<ClusterStatus> getAllClusterStatus();
}
