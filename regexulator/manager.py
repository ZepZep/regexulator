import math
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict, fields
from typing import Optional

import pandas as pd

from .config import RegexulatorConfig
from .explorer import RegexulatorExplorer
from .evaluator import eval_pattern 

class Regexulator:
    def __init__(self, cfg:Optional[RegexulatorConfig]=None, **kwargs):
        if cfg is None:
            cfg = RegexulatorConfig(**kwargs)
        elif kwargs:
            cfg_d = asdict_non_recursive(cfg)
            cfg_d.update(kwargs)
            cfg = RegexulatorConfig(**cfg_d)
            
        self.cfg = cfg

        self.cur_random = cfg.random_state if cfg.random_state is not None else random.randint(0, 1000000)

        self.fit_called = False

    def fit(self, dataset):
        if self.fit_called:
            raise Exception("fit was already called")
        self.fit_called = True
        
        cfg = self.cfg
        self.start_time = time.time()
        self.timeout_time = None if cfg.time_limit is None else self.start_time + cfg.time_limit
        
        self.train, self.val, self.test =  self.split_dataset(dataset, cfg.random_state)
        self.random_lock = threading.Lock()
        self.eval_lock = threading.Lock()
        self.print_lock = threading.Lock()

        self.stop_event = threading.Event()

        self.best_explorer = None
        self.pattern = None
        self.metrics_train = None
        self.metrics_val = None
        self.metrics_test = None
        
        self.task_queue = queue.PriorityQueue()
        self.initial = []
        initial_priority = 10
        with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
            try:
                # print("exec start")
                futures = []
                for wid in range(cfg.workers):
                    futures.append(executor.submit(self.work, wid))
                # print("exec submitted")
        
                for i in range(cfg.initial_splits):
                    explorer = RegexulatorExplorer(
                        self.train, "start", cfg,
                        name=str(i), random_state=self.next_rs()
                    )
                    explorer.depth = 1
                    explorer.priority = initial_priority
                    self.initial.append(explorer)
                    self.task_queue.put((-initial_priority, explorer))
                # print("exec initial put")


                self.task_queue.join()
                # print("exec joined")

            except Exception as e:
                print(f"Exception in Manager: {e}")
                self.stop_event.set()
                raise

            for _ in range(cfg.workers):
                self.task_queue.put((-100, "exit"))

            for future in as_completed(futures):
                result = future.result()
    

            # print("exec end")           

        # print("exec exited")
        self.end_time = time.time()
        self.time = self.end_time - self.start_time
        return self.pattern

    def work(self, wid):
        # with self.print_lock: print(f"{wid=} started")
        while True:
            try:
                priority, explorer = self.task_queue.get(timeout=1)

            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue
            
            if explorer == "exit":
                # with self.print_lock: print(f"{wid=} got exit")
                break

            # with self.print_lock: print(f"{wid=} got explorer {explorer.name}")
            
            if self.timeout_time is not None and time.time() > self.timeout_time:
                # with self.print_lock: print(f"{wid=} got timeout ({explorer.name=})")
                explorer.state = "failed"
                reason = "canceled: timeout"
                explorer.fail_reason = str(reason)
                self.task_queue.task_done()
                empty_queue(self.task_queue, reason)
                break

            # with self.print_lock: print(f"{wid=} run explorer {explorer.name}")
            try:
                explorer.run()
                self.post_explorer(explorer)
            except Exception as e:
                with self.print_lock: print(f"EXCEPTION in worker {wid=}: {repr(e)}")
                self.task_queue.task_done()
                explorer.fail(e)
                empty_queue(self.task_queue, "canceled: other exception")
                raise
            
            self.task_queue.task_done()
            # with self.print_lock: print(f"{wid=} fin explorer {explorer.name}")

    def post_explorer(self, explorer):
        self.eval_explorer(explorer)
        if explorer.depth >= self.cfg.max_depth:
            return
        if explorer.metrics_val is not None:
            parent_metric = explorer.metrics_val[self.cfg.primary_metric]
        else:
            parent_metric = 0
        for i in range(self.cfg.branching_factor):
            child = RegexulatorExplorer(
                self.train, "improve", self.cfg,
                name=f"{explorer.name}_{i}",
                initial_pattern=explorer.pattern,
                random_state=self.next_rs(),
            )
            child.depth = explorer.depth + 1
            explorer.children.append(child)
            child.parent = explorer
            priority = self.get_priority(parent_metric, explorer.depth, i)
            child.priority = priority
            self.task_queue.put((-priority, child))
    
    def next_rs(self):
        with self.random_lock:
            rs = self.cur_random
            self.cur_random += 1
        return rs

    def eval_explorer(self, explorer, metric=None):
        if self.val:
            new_metrics = eval_pattern(self.val, explorer.pattern)
        else:
            # use train metric
            new_metrics = explorer.metrics_train
        explorer.metrics_val = new_metrics
        if metric is None:
            metric = self.cfg.primary_metric
        if new_metrics is None:
            # self.log.events.append(("info", f"pattern_improvement: new pattern not compilable: `{new_pattern}`"))
            return
        with self.eval_lock:
            best_metrics = None if self.best_explorer is None else self.best_explorer.metrics_val
            if best_metrics is None or new_metrics[metric] > best_metrics[metric]:
                # if self.metrics_train is None:
                #     self.log.events.append(("info", f"pattern_improvement: new pattern is compilable: {metric}={new_metrics[metric]:.3f} `{new_pattern}`"))
                # else:
                #     self.log.events.append(("info", f"pattern_improvement: new pattern better at {metric}: {new_metrics[metric]:.3f} > {self.metrics_train[metric]:3f} `{new_pattern}`"))
                self.best_explorer = explorer
                self.pattern = explorer.pattern
            else:
                pass
            # self.log.events.append(("info", f"pattern_improvement: new pattern worse at {metric}: {new_metrics[metric]:.3f} < {self.metrics_train[metric]:3f} `{new_pattern}`"))
            
        

    def get_priority(self, metric, depth, n_siblings):
        p = metric
        p *= self.cfg.priority_depth_multiplier ** depth
        p *= self.cfg.priority_sibling_multiplier ** n_siblings
        return p
        

    def node_apply(self, fcn, node=None):
        if node is None:
            for node in self.initial:
                self.node_apply(fcn, node)
        else:
            fcn(node)
            for child in node.children:
                self.node_apply(fcn, child)

    def to_dot(self):
        parts = ["digraph G {\n  rankdir=LR\n"]
        def add_node_part(node):
            parts.append(self.node_to_dot(node))
        self.node_apply(add_node_part)
        parts.append("}")
        return "\n".join(parts)

    def show_dot(self):
        import graphviz
        return graphviz.Source(self.to_dot())

    def to_dict(self):
        return {
            "cfg": self.cfg._asdict(),
            "time": self.time,
            "best_explorer": None if self.best_explorer is None else self.best_explorer.to_dict(children=False),
            "best_explorer_id": id(self.best_explorer),
            "pattern": self.pattern,
            "metrics_train": self.metrics_train,
            "metrics_val": self.metrics_val,
            "metrics_test": self.metrics_test,
            "dot": self.to_dot(),
            "nodes": [explorer.to_dict(children=True) for explorer in self.initial],
        }
        pass

    def split_dataset(self, dataset, random_state=None):
        cfg = self.cfg
        dataset = list(dataset)
        if random_state is not None:
            random.Random(random_state).shuffle(dataset)
        l = len(dataset)
        val_len = math.floor(cfg.validation_fraction * l)
        test_len = math.floor(cfg.test_fraction * l)
        train_len = l - val_len - test_len

        # print(f"{train_len=}, {val_len=}, {test_len=}")

        train = dataset[:train_len]
        val = dataset[train_len:train_len+val_len]
        test = dataset[train_len+val_len:]
        return train, val, test


    def node_to_dot(self, node):
        pattern = "" if node.pattern is None else record_escape(node.pattern)
        parent = "root" if node.parent is None else node.parent.name
        if node.state == "finished":
            tm = "-" if node.metrics_train is None else node.metrics_train[self.cfg.primary_metric]
            vm = "-" if node.metrics_val is None else node.metrics_val[self.cfg.primary_metric]
            fillcolor = get_color_scale(vm, BG_SCALE_METRIC)
            label = f"{pattern}|{{{{{'train_'+self.cfg.primary_metric}|{'val_'+self.cfg.primary_metric}}}|{{{m_to_str(tm)}|{m_to_str(vm)}}}|{{not_compile|self_revise}}|{{{node.log.n_not_compilable}|{node.log.n_self_revised}}}}}"
            return f"""  "{node.name}" [shape=record, style=filled fillcolor="{fillcolor}" label="{label}"]
  "{parent}" ->  "{node.name}"
"""
        else:
            label = f"{pattern}|state: {node.state}|{record_escape(str(node.fail_reason))}"
            return f""""{node.name}" [shape=record, style=filled fillcolor="#E99999" label="{label}"]
  "{parent}" ->  "{node.name}"
"""

def m_to_str(m):
    if type(m) == str:
        return m
    return f"{m:.3f}"

def record_escape(text):
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\n").replace("<", "\\<").replace(">", "\\>")

BG_SCALE_METRIC = [
    (0.4, "#F6B26B"),
    (0.5, "#F9CA9C"),
    (0.6, "#FCE4CC"),
    (0.7, "#D9E9D3"),
    (0.8, "#B6D7A7"),
    (0.9, "#93C37D"),
    (1.0, "#6AA74F"),
]
def get_color_scale(val, scale):
    if type(val) == str:
        return scale[0][1]
    for tr, color in scale:
        if tr > val:
            return color
    return color
    
def ds_stats():
    pass

def empty_queue(q, reason=None):
    try:
        while True:
            _, explorer = q.get(block=False)
            explorer.state = "failed"
            explorer.fail_reason = str(reason)
            q.task_done()
    except queue.Empty:
        pass
            
def asdict_non_recursive(instance):
    return {f.name: getattr(instance, f.name) for f in fields(instance)}
    