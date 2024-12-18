import time
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


class NonOverlappingTimeProfiler(object):
    def __init__(self):
        self.time_cost = defaultdict(float)
        self.tic = time.time()

    def end(self, key):
        toc = time.time()
        self.time_cost[key] += toc - self.tic
        self.tic = toc

    def reset(self):
        self.time_cost.clear()
        self.tic = time.time()

    def read(self):
        tot_time = sum(self.time_cost.values())
        ratio = {f'{k}_ratio': v / tot_time for k, v in self.time_cost.items()}
        return {**self.time_cost, **ratio, **{'total': tot_time}}
    
    def dump_to_writer(self, writer: SummaryWriter, global_step):
        time_stat = self.read()
        writer.add_scalar("time/SPS", global_step / time_stat.pop('total'), global_step)
        for k, v in time_stat.items():
            if k.endswith('ratio'):
                writer.add_scalar(f"time/{k}", v, global_step)
            else:
                writer.add_scalar(f"time/{k}_SPS", global_step / v, global_step)