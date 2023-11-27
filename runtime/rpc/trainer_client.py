import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../rpc_stubs'))

from runtime.rpc_stubs.trainer_to_scheduler_pb2 import ReportStatsRequest, ReportStatsResponse, RegisterTrainerRequest
import runtime.rpc_stubs.trainer_to_scheduler_pb2_grpc as t2s_rpc

import grpc


class TrainerClientForScheduler(object):
    def __init__(self, logger, scheduler_ip, scheduler_port) -> None:
        super().__init__()

        self._logger = logger

        self._scheduler_ip = scheduler_ip
        self._scheduler_port = scheduler_port

        # channel = grpc.insecure_channel(self.addr)
        # self._stub = t2s_rpc.TrainerToSchedulerStub(channel)
    

    @property
    def addr(self):
        return f'{self._scheduler_ip}:{self._scheduler_port}'
    

    # def register_trainer(self, trainer_ip, trainer_port, job_id):
    #     request = RegisterTrainerRequest(trainer_ip=trainer_ip, trainer_port=trainer_port, job_id=job_id)

    #     try:
    #         response = self._stub.RegisterTrainer(request)
    #         self._logger.info(f'job {job_id}, register, {response.success}')
    #         return response.success
    #     except Exception as e:
    #         self._logger.info(f'job {job_id}, register, fail, {e}')
    #         return False, None
    

    def report_stats(self, job_id, finished_iterations):
        try:
            self._logger.info(f'job {job_id}, report, {finished_iterations}')
            with grpc.insecure_channel(self.addr) as channel: # with 语句确保在代码块结束后，通道被正确关闭。
                request = ReportStatsRequest(job_id=job_id, finished_iterations=finished_iterations)
                # 向 scheduler 发送 ReportStatsRequest 请求
                stub = t2s_rpc.TrainerToSchedulerStub(channel)
                # 这个stub是用于发起RPC调用的客户端对象
                response = stub.ReportStats(request)
                # 通过stub发送RPC请求，并等待响应。
                assert response.success == True
        except Exception as e:
            self._logger.info(f'job {job_id}, report, fail, {e}')
