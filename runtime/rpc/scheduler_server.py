import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))


from runtime.rpc_stubs.trainer_to_scheduler_pb2 import ReportStatsRequest, ReportStatsResponse, RegisterTrainerRequest, RegisterTrainerResponse
from runtime.rpc_stubs.trainer_to_scheduler_pb2_grpc import TrainerToSchedulerServicer
import runtime.rpc_stubs.trainer_to_scheduler_pb2_grpc as t2s_rpc

import grpc
from concurrent import futures


class SchedulerServerForTrainer(TrainerToSchedulerServicer):
    def __init__(self, logger, callbacks) -> None:
        super().__init__()

        self._logger = logger
        self._callbacks = callbacks
    

    def RegisterTrainer(self, request: RegisterTrainerRequest, context):
        # return super().RegisterTrainer(request, context)
        assert 'RegisterTrainer' in self._callbacks
        register_trainer_impl = self._callbacks['RegisterTrainer']

        success = register_trainer_impl(request.trainer_ip, request.trainer_port, request.job_id)
        response = RegisterTrainerResponse(success=success)

        return response
    

    def ReportStats(self, request: ReportStatsRequest, context) -> ReportStatsResponse:
        # return super().ReportStats(request, context)
        assert 'ReportStats' in self._callbacks
        report_stats_impl = self._callbacks['ReportStats'] # 从callbacks这个字典中取出对应的rpc业务逻辑实现

        success = report_stats_impl(request.job_id, request.finished_iterations) # 调用对应的实现
        response = ReportStatsResponse(success=success) # 将report_stats_impl返回的布尔值结果封装为ReportStatsResponse
        
        return response


def serve(port, logger, callbacks):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10)) # 允许服务器同时处理多个请求
    t2s_rpc.add_TrainerToSchedulerServicer_to_server(SchedulerServerForTrainer(logger, callbacks), server)
    # t2s_rpc 是由 trainer_to_scheduler.proto 文件生成的，调用该函数是将一个服务接口实现（在这个例子中是 SchedulerServerForTrainer 类的实例）注册到 gRPC 服务器上
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    logger.info(f'worker, rpc, start, server @ {port}')
    
    # server.wait_for_termination()
    return server