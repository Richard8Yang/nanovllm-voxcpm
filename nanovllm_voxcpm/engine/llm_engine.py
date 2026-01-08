import atexit
import torch.multiprocessing as mp

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.sequence import Sequence
from nanovllm_voxcpm.engine.scheduler import Scheduler
from nanovllm_voxcpm.engine.model_runner import RunnerTask, BaseModelRunner
import socket
import torch

def get_distributed_port():
    # find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class LLMEngineBase:
    model_runner : BaseModelRunner
    scheduler : Scheduler
    
    def __init__(self, RunnerType : type[BaseModelRunner], config: Config, tensor_parallel_size: int):
        
        self.distributed_port = get_distributed_port()

        if config.devices is None or len(config.devices) == 0:
            n_devices = torch.cuda.get_device_count()
            if tensor_parallel_size > n_devices:
                raise ValueError(f"Tensor parallel size {tensor_parallel_size} is greater than the number of available devices {n_devices}")
            config.devices = list(range(tensor_parallel_size))

        if len(config.devices) != tensor_parallel_size:
            raise ValueError(f"Number of devices {len(config.devices)} is not equal to tensor parallel size {tensor_parallel_size}")

        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=RunnerType, args=(config, i, config.devices[i], self.distributed_port, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = RunnerType(config, 0, config.devices[0], self.distributed_port, self.events)
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_sequence(self, seq : Sequence):
        self.scheduler.add(seq)
    
    def cancel_sequence(self, seq_id: str):
        self.scheduler.cancel(seq_id)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        runner_tasks = [self.preprocess_seq(seq, is_prefill) for seq in seqs]
        
        if not runner_tasks:
            return []
        
        outputs = self.model_runner.call("run", runner_tasks, is_prefill)
        
        for seq, output in zip(seqs, outputs):
            self.postprocess_seq(seq, output, is_prefill)
        
        for seq in seqs:
            if seq.stoped:
                self.scheduler.finish(seq)

        return seqs

    def is_finished(self):
        return self.scheduler.is_finished()
    
    def preprocess_seq(self, seq : Sequence, is_prefill: bool) -> RunnerTask:
        raise NotImplementedError()
    
    def postprocess_seq(self, seq : Sequence, outputs : dict, is_prefill: bool):
        raise NotImplementedError()
