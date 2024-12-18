import random
import regex
from dataclasses import dataclass, field, asdict

import ollama
import pandas as pd

from .evaluator import eval_pattern, get_eval_examples
from .config import RegexulatorConfig


class RegexulatorExplorer:
    def __init__(self, train, task_type, cfg: RegexulatorConfig, name, random_state=None,
                 initial_pattern=None, validation_feedback=None):
        self.state = "init"
        
        self.train=train
        self.task_type = task_type
        self.cfg = cfg
        self.random_state = random_state if random_state is not None else random.randint(0, 1000000)

        self.initial_pattern = initial_pattern
        self.validation_feedback = validation_feedback

        # tree info
        self.name = name
        self.priority = None
        self.depth = None
        self.parent = None
        self.children = []

        # results
        self.pattern = None
        self.metrics_train = None
        self.metrics_val = None
        
        # "queued" | "running" | "interrupted" | "finished" | "failed"
        self.fail_reason = None
        self.log = RegexulatorExplorerLog()
        # "time, time_breakdown, prompts, n_self_revised, n_not_compilable"

        if len(train) == 0:
            raise ValueError("len(train) == 0")
        if "string" in train[0]:
            self.text_name = "string"
        elif "text" in train[0]:
            self.text_name = "text"
        else:
            raise ValueError("Elements of train should have `text` or `string` key.")

    def __lt__(self, other):
        return False

    def run(self):
        self.state = "running"
        if self.task_type == "start":
            prompt = self.make_start_prompt()
        elif self.task_type == "improve":
            prompt = self.make_improve_prompt()

        messages = messages = [msg_user(prompt)]
        name = f"{self.task_type}"
        response = self.llm_generate(name, messages)
        messages.append(msg_assistant(response))
        pattern = self.extract_regex(response)

        self.pattern = pattern
        self.metrics_train = eval_pattern(self.train, pattern, self.cfg)
        
        if self.metrics_train is None:
            self.log.events.append(("info", f"first_guess: new pattern not compilable: `{pattern}`"))
        else:
            self.log.events.append(("info", f"first_guess: pattern ok: {self.cfg.primary_metric}={self.metrics_train[self.cfg.primary_metric]:.3f} `{pattern}`"))
        
        passed = 0
        while True:
            if self.log.n_not_compilable < self.cfg.max_check_compile:
                skipped, new_pattern = self.check_compile(messages, pattern)
                if not skipped:
                    pattern = new_pattern
                    self.pattern_improvement(pattern)
                    continue
            if self.log.n_self_revised < self.cfg.max_self_revise:
                skipped, new_pattern = self.self_revise(messages)
                if not skipped:
                    pattern = new_pattern
                    self.pattern_improvement(pattern)
                    continue
            break

        
        self.state = "finished"

    def make_start_prompt(self):
        train_flat = self._flatten_ds(self.train)
        to_sample = self.cfg.show_start_examples
        if len(train_flat) == 0:
            raise Exception("No start examples")
        if len(train_flat) < self.cfg.show_start_examples:
            to_sample = len(train_flat)
            self.log.events.append(("warn", f"too few start examples: {len(train_flat)} < {self.cfg.show_start_examples} (cfg.show_start_examples)"))

        selected = train_flat.sample(to_sample, random_state=self.random_state)
        examples, extractions = self.make_exa_ext(selected)


        prompt = self.cfg.prompt_templates.start.format(
            examples=examples,
            extractions=extractions,
            cheatsheet=self.get_cheatsheet(),
            description=self.get_description(),
        )
        return prompt

    def make_improve_prompt(self):
        if self.initial_pattern is None:
            raise Exception("Task type is improve but initial pattern is None")
        train_flat = get_eval_examples(self.train, self.initial_pattern, self.cfg)
        # print(f"improve vc: {train_flat.type.value_counts().to_dict()}", flush=True)

        tp = self.select_for_improve(train_flat, "exact")
        fn = self.select_for_improve(train_flat, "miss_gt")
        fp = self.select_for_improve(train_flat, "miss_pred")

        prompt = self.cfg.prompt_templates.improve.format(
            pattern=self.initial_pattern,
            examples_tp=tp[0], extractions_tp=tp[1],
            examples_fn=fn[0], extractions_fn=fn[1],
            examples_fp=fp[0], extractions_fp=fp[1],
            cheatsheet=self.get_cheatsheet(),
            description=self.get_description(),
        )
        return prompt

    def select_for_improve(self, flat, subset):
        sub = flat.query(f"type == '{subset}'")
        if len(sub) == 0:
            return "", ""
        to_sample = min(self.cfg.show_improve_examples, len(sub))
        selected = sub.sample(to_sample, random_state=self.random_state)
        return self.make_exa_ext(selected)

    def fail(self, exception):
        # print("f1", flush=True)
        old_state = self.state
        if self.fail_reason:
            self.fail_reason += "\n"
        else:
            self.fail_reason = ""
        # print("f2", flush=True)
        self.fail_reason += f"exception in {old_state}: {repr(exception)}"
        self.state = "failed"
        # print("f3", flush=True)

    def get_cheatsheet(self):
        if not self.cfg.cheatsheet:
            return ""
        return self.cfg.prompt_templates.cheatsheet

    def get_description(self):
        if not self.cfg.natural_description:
            return ""
        return f"The expression should match {self.cfg.natural_description}\n"
        
    
    def pattern_improvement(self, new_pattern, metric=None):
        new_metrics = eval_pattern(self.train, new_pattern, self.cfg)
        if metric is None:
            metric = self.cfg.primary_metric
        if new_metrics is None:
            self.log.events.append(("info", f"pattern_improvement: new pattern not compilable: `{new_pattern}`"))
            return
        if self.metrics_train is None or new_metrics[metric] > self.metrics_train[metric]:
            if self.metrics_train is None:
                self.log.events.append(("info", f"pattern_improvement: new pattern is compilable: {metric}={new_metrics[metric]:.3f} `{new_pattern}`"))
            else:
                self.log.events.append(("info", f"pattern_improvement: new pattern better at {metric}: {new_metrics[metric]:.3f} > {self.metrics_train[metric]:3f} `{new_pattern}`"))
            self.pattern = new_pattern
            self.metrics_train = new_metrics
        else:
            self.log.events.append(("info", f"pattern_improvement: new pattern worse at {metric}: {new_metrics[metric]:.3f} < {self.metrics_train[metric]:3f} `{new_pattern}`"))
            

    def check_compile(self, messages, pattern):
        try:
            if not pattern:
                raise regex.error("No pattern found")
            regex.compile(pattern)
        except regex.error as e:
            print("check_compile triggered")
            self.log.n_not_compilable += 1
            prompt = self.cfg.prompt_templates.not_compilable.format(error=str(e))
            messages.append(msg_user(prompt))
            response = self.llm_generate(f"fix_compile-{self.log.n_not_compilable}", messages)
            messages.append(msg_assistant(response))
            pattern = self.extract_regex(response)
            return False, pattern
        
        return True, None

    def self_revise(self, messages):
        new_message = msg_user(self.cfg.prompt_templates.self_revise_question)
        response = self.llm_generate(f"self_revise_question-{self.log.n_self_revised}", messages+[new_message])
        if response.lower() == "yes":
            return True, None
        if response.lower() == "no":
            print("self_revise triggered")
            self.log.n_self_revised += 1
            prompt = self.cfg.prompt_templates.self_revise_fix
            messages.append(msg_user(prompt))
            response = self.llm_generate(f"self_revise_question-{self.log.n_self_revised}", messages)
            messages.append(msg_assistant(response))
            pattern = self.extract_regex(response)
            return False, pattern
        else:
            print(f"unknown self_revise response: {response}")

    def make_exa_ext(self, ds_flat):
        exa = []
        ext = []
        for _, row in ds_flat.iterrows():
            exa.append(self.make_example(
                row[self.text_name], row["start"], row["end"]
            ))
            ext.append(row[self.text_name][row["start"]:row["end"]])
        return "\n".join(exa), "\n".join(ext)

    def make_example(self, text, l, r):
        ll = self.find_stop(text, l, -1)
        rr = self.find_stop(text, r,  1)
        # FIXME tags
        return text[ll:l] + "`" + text[l:r] + "`" + text[r:rr]
    
    def find_stop(self, text, a, dir):
        aa = a
        if dir == -1:
            aa -= self.cfg.min_context_pre_chars
        aa = max(0, min(aa, len(text)-1))
        last_break = None
        while 0 < aa < len(text)-1 and abs(a - aa) < self.cfg.max_context_chars:
            if text[aa] in "\n\t ":
                last_break = aa-dir
                if dir == 1 and text[aa] == "\n":
                    return aa-dir
            aa += dir
    
        if last_break is None:
            return aa
        return last_break

    def llm_generate(self, name, messages):
        # print(f"GEN S {self.random_state}\n", flush=True, end="")
        e = ollama.chat(
            model=self.cfg.model,
            messages=messages,
            options={
                "num_ctx": self.cfg.llm_num_ctx,
                "num_predict": self.cfg.max_predict_tokens,
                "temperature": self.cfg.llm_temperature
            },
            keep_alive=self.cfg.llm_keep_alive,
        )
        self.log.conversations.append(make_conversation(name, messages, e))
        # print(f"GEN F {self.random_state}\n", flush=True, end="")
        return e['message']['content']

    def extract_regex(self, response):
        m = regex.search("FINAL REGEX:\s*(.*)", response)
        if m is None:
            # fixme ask for final regex
            return None
        # FIXME tags
        return m.group(1).replace("`", "")


    def _flatten_ds(self, ds):
        return pd.DataFrame.from_records([
            {self.text_name: doc[self.text_name], **ann}
                for doc in ds
                for ann in doc["match"]
        ])

    def to_dict(self, children=True):
        out = {}
        for atr in ['task_type', 'initial_pattern', 'validation_feedback',
                    'name', 'priority', 'depth', 'pattern', 'metrics_train',
                    'metrics_val', 'fail_reason']:
            out[atr] = self.__getattribute__(atr)
        out["log"] = self.log.to_dict()
        if children:
            out["children"] = [child.to_dict(children=children) for child in self.children]
        return out


def msg_user(prompt):
    return {
      "role": "user",
      "content": prompt
    }

    
def msg_assistant(prompt):
    return {
      "role": "assistant",
      "content": prompt
    }

    
@dataclass
class RegexulatorExplorerLog:
    time: float | None = None
    events: list = field(default_factory=list)
    conversations: list = field(default_factory=list)
    n_not_compilable: int = 0
    n_self_revised: int = 0

    def to_dict(self):
        return asdict(self)

@dataclass
class RegexulatorConversation:
    name: str
    prompt: list
    response: dict
    model: str
    created_at: str
    done_reason: str
    done: bool
    prompt_eval_count: int
    eval_count: int
    total_duration: float
    load_duration: float
    prompt_eval_duration: float
    eval_duration: float

    def print(self):
        for d in self.prompt + [self.response]:
            print(f"\n{d['role']:-^20}\n")
            print(d["content"])

        # print(f"\n{'assistant':-^20}\n")
        # print(self.response)

def make_conversation(name, prompt, response):
    stay = ["model", "created_at", "done_reason", "done", "prompt_eval_count", "eval_count"]
    to_float = ['total_duration', 'load_duration', 'prompt_eval_duration', 'eval_duration']
    transformed = {
        "name": name,
        "prompt": prompt.copy(),
        "response": response["message"],
        ** {k:response[k] for k in stay},
        ** {k:response[k]/1e9 for k in to_float},
    }
    return RegexulatorConversation(**transformed)
    


    