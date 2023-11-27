import argparse

# from sympy import false
import utils
import threading
import time
import subprocess
import os
import pynvml
import csv
import queue

from runtime.rpc import scheduler_server
from task import Task, JobInfo


class Worker(object):
    # 初始化Worker对象，接收：trace文件路径、工作节点IP和端口、GPU资源、挂载点、日志路径等参数
    # 其中，trace文件路径指定了trace配置文件（如test_tgs.csv），用于提取任务信息。
    # 工作节点IP和端口用于创建RPC服务器，以接收来自调度器的RPC请求。
    # GPU资源指定了该工作节点可用的GPU资源。
    # 挂载点指定了该工作节点可用的挂载点。
    def __init__(self, trace_file_path: str, worker_ip, worker_port, gpus: str, mount: list, log_path: str, need_throughput) -> None:
        super().__init__()

        self._logger = utils.make_logger(__name__)
        self._writer = utils.Writer(log_path)

        self.parse_trace_config(trace_file_path) # 提取低优先级与高优先级的任务信息
        
        self._worker_ip = worker_ip
        self._worker_port = worker_port
        self._worker_id = None
        self.need_throughput = need_throughput
        
        self._gpus = gpus.split(',') # 该工作节点可用的GPU资源，如'0,1,2,3'，默认为0号GPU
        self._num_gpus = len(self._gpus)

        self._mount = mount if mount != None else []

        self.tgs_init() # 编译拦截库
        
        self._tasks = dict()

        self._server_for_trainer = self.make_server_for_trainer(worker_port) # 创建并启动 RPC 服务

        self._start_time = time.time() # 记录worker启动时间
    
    # 解析trace配置文件（如test_tgs.csv），可能用于提取任务信息。
    # trace配置文件中的每一行都是一个作业的规范，包括：提交时间、模型名称、批大小、迭代次数、是否有GPU需求、优先级、Docker镜像名称等。
    def parse_trace_config(self, trace_file_path):
        assert trace_file_path[-4:] == '.csv'
        trace_file = open(trace_file_path, 'r')

        reader = csv.DictReader(trace_file, delimiter=',', skipinitialspace=True)

        self._submit_queue = list()
        self.next_job_id = 1
        for row in reader:
            self.parse_job(row) 
        
        trace_file.close()
        self._submit_queue = sorted(self._submit_queue, key=lambda x: (x['submit_time'], 0 if x['priority'] == 'high' else 1)) # 先按提交时间排序，再按优先级排序

    # 解析单个作业的具体信息。
    def parse_job(self, job_spec):
        assert 'submit_time' in job_spec
        assert 'model_name' in job_spec
        assert 'batch_size' in job_spec
        assert 'iterations' in job_spec
        assert 'gpu_requests' in job_spec
        assert 'priority' in job_spec

        # if job_spec['model_name'] == 'shufflenet':
        #     job_spec['model_name'] = 'shufflenet_v2_x1_0'

        spec = {
            'submit_time': float(job_spec['submit_time']),
            'job_id': self.next_job_id,
            'model_name': job_spec['model_name'],
            'batch_size': job_spec['batch_size'],
            'iterations': int(job_spec['iterations']),
            'num_gpus': int(job_spec['gpu_requests']),
            'priority': job_spec['priority'],
            'thread_percentage': job_spec['thread_percentage'] if 'thread_percentage' in job_spec else None,
            'image_name': job_spec['image_name'] if 'image_name' in job_spec else 'tf_torch',
            'antman_config': job_spec['antman_config'] if 'antman_config' in job_spec else None,
            'antman_status': job_spec['antman_status'] if 'antman_status' in job_spec else None,
        }
        
        self._submit_queue.append(spec) # _submit_queue 用于存储所有作业的具体信息
        self.next_job_id += 1

    # 用于初始化GPU共享或其他任务调度相关的设置。
    # 实际上是调用了hijack/build.sh脚本，编译拦截库，并分别复制到hijack/high-priority-lib和hijack/low-priority-lib目录下。
    def tgs_init(self):
        assert subprocess.call(['./hijack/build.sh']) == 0
        root_path = os.path.abspath('.')

        self.tgs_mounts = { # 建立宿主机上的路径和容器内的路径之间的映射
            'high': [
                root_path + ':/cluster',
                root_path + '/hijack/high-priority-lib/libcontroller.so:/libcontroller.so:ro',
                root_path + '/hijack/high-priority-lib/libcuda.so:/libcuda.so:ro',
                root_path + '/hijack/high-priority-lib/libcuda.so.1:/libcuda.so.1:ro',
                root_path + '/hijack/high-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro',
                root_path + '/hijack/high-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro',
                root_path + '/hijack/high-priority-lib/ld.so.preload:/etc/ld.so.preload:ro',
                root_path + '/gsharing:/etc/gsharing',
            ],
            'low': [
                root_path + ':/cluster',
                root_path + '/hijack/low-priority-lib/libcontroller.so:/libcontroller.so:ro',
                root_path + '/hijack/low-priority-lib/libcuda.so:/libcuda.so:ro',
                root_path + '/hijack/low-priority-lib/libcuda.so.1:/libcuda.so.1:ro',
                root_path + '/hijack/low-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro',
                root_path + '/hijack/low-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro',
                root_path + '/hijack/low-priority-lib/ld.so.preload:/etc/ld.so.preload:ro',
                root_path + '/gsharing:/etc/gsharing',
            ],
            'Ex': [
                root_path + ':/cluster',
            ],
            'Co-ex': [
                root_path + ':/cluster',
            ],
            'mig-high': [
                root_path + ':/cluster',
            ],
            'mig-low': [
                root_path + ':/cluster',
            ],
            'mps': [
                root_path + ':/cluster',
                '/tmp/nvidia-mps:/tmp/nvidia-mps',
            ],
        }

    # 从 _task 中检查是否有作业已经完成，如果有，则将其从 _task 中移除，并将其记录到日志中，同时返回。
    def check_tasks(self):
        finished_tasks = []

        for job_id, task in self._tasks.items():
            if task.return_code == None:
                continue
            assert task._finished_iterations == task._iterations
            
            finished_tasks.append(task)
        
        if len(finished_tasks) > 0:
            self.record()
        for task in finished_tasks:
            self._tasks.pop(task._job_id)
        
        return finished_tasks
    
    # 执行一个作业，可能包括在容器中运行深度学习模型。
    def execute(self, job_info) -> bool:
        success = True
        # 作业转换为 worker 的任务
        task = Task(job_info, self._worker_ip, self.tgs_mounts, self.need_throughput)
        # worker 的 _tasks 列表中记录该作业 id 所对应的任务类实例
        self._tasks[task._job_id] = task
        cmd = task.run(self._mount)

        self._logger.info(f'{self._worker_id}, execute, {task._job_id}, {task._gpus}, {task._priority}, {" ".join(cmd)}')

        return success
    
    # 终止一个正在运行的作业。
    def kill(self, job_info) -> bool:
        job_id = job_info.job_id

        if job_id not in self._tasks:
            return False

        task = self._tasks.pop(job_id)
        task.terminate()

        self._logger.info(f'{self._worker_id}, kill, {job_id}, {job_info.gpus}, {job_info.priority}')

        return True
    
    # 查询节点（如GPU）的统计信息。
    def query_node_stats(self):
        utilizations = []
        pynvml.nvmlInit()
        for gpu_id in range(self._num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            utilizations.append(str(utilization))
        pynvml.nvmlShutdown()

        self._logger.info(f'{self._worker_id}, query, {"-".join(utilizations)}')
        utilizations = ','.join(utilizations)
        return utilizations

    # ReportStats RPC 对应的具体业务逻辑，报告作业的状态，可能包括完成的迭代次数等。
    def _report_stats_impl(self, job_id, finished_iterations) -> bool:
        success = True
        assert job_id in self._tasks
        task = self._tasks[job_id]
        throughput = task.update(finished_iterations)

        self._logger.info(f'worker, report, {job_id}, {throughput}, {task._finished_iterations}')

        return success


    def make_server_for_trainer(self, port):
        callbacks = {
            'ReportStats' : self._report_stats_impl,
        } # 当 ReportStats RPC调用发生时，应该调用 _report_stats_impl 方法处理请求

        return scheduler_server.serve(port, self._logger, callbacks) # 创建并启动 RPC 服务，该服务将作为 worker (或者说，scheduler )和 trainer 之间的通信桥梁


    def has_ready_jobs(self):
        current_time = time.time()
        elapsed_time = current_time - self._start_time # worker 的运行时间

        if len(self._submit_queue) > 0:
            job_spec = self._submit_queue[0] # 检查队首（实际上就是list的首个元素）的作业
            if job_spec['submit_time'] <= elapsed_time: # 注意到 submit_time 一般是 0
                return True
        
        return False


    def record(self):
        timestamp = time.time() - self._start_time
        for task in self._tasks.values():
            task.record(timestamp, self._writer)


    def close(self):
        self._writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_port', type=int, default=6889)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--mount', action='append')
    parser.add_argument('--trace', type=str,  required=True) # default='config/test_tgs.csv')
    parser.add_argument('--log_path', type=str,  required=True) # default='results/test_tgs_results.csv')
    parser.add_argument('--need_throughput', action='store_true', default=False)
    args = parser.parse_args()

    subprocess.call('docker stop $(docker ps -q)', shell=True)
    subprocess.call('docker rm $(docker ps -aq)', shell=True)

    worker_ip = utils.get_host_ip() # 获取本机IP地址
    worker = Worker(args.trace, worker_ip, args.worker_port, args.gpus, args.mount, args.log_path, args.need_throughput)

    runnable_tasks = list()
    gpu_list = args.gpus.split(',')
    machine = [{
        'Co-ex': list(),
        'mps': list()
    } for i in range(len(gpu_list))] # 用于记录每个GPU上正在运行的作业，以及对应的模式或者优先级，对于 TGS 而言，是 high 或者 low
    while len(worker._submit_queue) + len(worker._tasks) + len(runnable_tasks) > 0: # 定期检查需要提交的作业以及正在运行的作业
        while worker.has_ready_jobs():
            job_spec = worker._submit_queue.pop(0)
            jobinfo = JobInfo(job_spec['job_id'], job_spec['model_name'], job_spec['batch_size'],
                 job_spec['iterations'], job_spec['num_gpus'], job_spec['priority'],
                 job_spec['thread_percentage'], job_spec['image_name'],
                 job_spec['antman_config'], job_spec['antman_status']
                )
            runnable_tasks.append(jobinfo) # 从 _submit_queue 中取出作业，放入 runnable_tasks 中

        finished_tasks = worker.check_tasks()
        for task in finished_tasks:
            for gpu_id in task._gpus.split(','):
                if task._priority in ['Co-ex', 'mps']:
                    machine[int(gpu_id)][jobinfo.priority].remove(task._job_id)
                else:
                    machine[int(gpu_id)].pop(task._priority) # 将该优先级的任务从 machine(即 GPU) 中移除
            # writer.save(task)
        
        new_runnable_tasks = []
        record_flag = (len(finished_tasks) != 0)
        for jobinfo in runnable_tasks: # 将 runnable_tasks 中的作业分配给可用的 GPU
            available_gpus = 0
            for gpu_instance in machine:
                if jobinfo.priority not in gpu_instance: 
                    available_gpus += 1
                elif jobinfo.priority in ['Co-ex', 'mps'] and len(gpu_instance[jobinfo.priority]) < 2:
                    available_gpus += 1
            
            if available_gpus >= jobinfo.num_gpus: # 如果可用的 GPU 数量大于等于作业所需的 GPU 数量，则将该作业分配给可用的 GPU
                record_flag = True
                used_gpus = [] # 用于记录该作业所使用的 GPU
                for gpu_id, gpu_instance in enumerate(machine): # 
                    if jobinfo.priority not in gpu_instance:
                        used_gpus.append(str(gpu_id)) 
                        gpu_instance[jobinfo.priority] = jobinfo.job_id # 在该 machine 中记录该优先级下的作业 id
                    elif jobinfo.priority in ['Co-ex', 'mps'] and len(gpu_instance[jobinfo.priority]) < 2:
                        used_gpus.append(str(gpu_id))
                        gpu_instance[jobinfo.priority].append(jobinfo.job_id)
                    
                    if len(used_gpus) == jobinfo.num_gpus:
                        break
                jobinfo.gpus = ','.join(used_gpus) # 该作业所使用的 GPU 列表
                worker.execute(jobinfo)
            else: # 可用的 GPU 数量不够，先放到 new_runnable_tasks 中
                new_runnable_tasks.append(jobinfo)

        if record_flag:
            worker.record()
        runnable_tasks = new_runnable_tasks

        sleep_time = 2
        if len(worker._submit_queue) > 0:
            sleep_time = min(sleep_time, (worker._start_time + worker._submit_queue[0]['submit_time'] - time.time()))
        time.sleep(sleep_time)
    
    worker.close()