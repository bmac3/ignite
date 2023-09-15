from collections import defaultdict
import functools

from .events import *


class TerminateEpochException(Exception):
    pass


class TerminateRunException(Exception):
    pass


class FilterEventException(Exception):
    pass


def is_special_var(key):
    return key.startswith('__') and key.endswith('__')


def update_step(state):
    state.step += 1
    return state


def update_iteration(state):
    state.iteration += 1
    return state


def update_epoch(state):
    state.epoch += 1
    state.iteration = 0
    return state


class Engine:
    
    event_to_attr = {
        Events.GET_BATCH_STARTED: "step",
        Events.GET_BATCH: "step",
        Events.GET_BATCH_COMPLETED: "step",
        Events.ITERATION_STARTED: "step",
        Events.PROCESS: "step",
        Events.ITERATION_COMPLETED: "step",
        Events.CHECKPOINT: "step",
        Events.EPOCH_STARTED: "epoch",
        Events.EPOCH_COMPLETED: "epoch",
        Events.STARTED: "epoch",
        Events.TERMINATE: "epoch",
        Events.COMPLETED: "epoch",
    }
    
    def __init__(self):
        self.event_handlers = defaultdict(list)
        self.add_event_handler(Events.ITERATION_COMPLETED, update_step, ('state',), ('state',))
        self.add_event_handler(Events.ITERATION_COMPLETED, update_iteration, ('state',), ('state',))
        self.add_event_handler(Events.EPOCH_COMPLETED, update_epoch, ('state',), ('state',))
        
    def add_event_handler(self, event_name, handler, input_args=(), output_args=()):
        if not isinstance(input_args, (tuple, list)):
            input_args = (input_args,)
        if not isinstance(output_args, (tuple, list)):
            output_args = (output_args,)
        if isinstance(event_name, EventsList):
            for e in event_name:
                self.add_event_handler(e, handler, input_args, output_args)
        else:
            if isinstance(event_name, CallableEventWithFilter) and event_name.filter is not None:
                event_filter = event_name.filter
                handler = self.handler_wrapper(handler, event_name, event_filter)
            self.event_handlers[event_name].append((handler, input_args, output_args))
            
    def handler_wrapper(self, handler, event_name, event_filter):
        @functools.wraps(handler)
        def wrapper(*args, **kwargs):
            event = self.get_event_attrib_value(event_name)
            if event_filter(self, event):
                return handler(*args, **kwargs)
            else:
                raise FilterEventException()
            
        return wrapper
    
    def get_event_attrib_value(self, event_name):
        attr = self.event_to_attr[event_name]
        return getattr(self.state, attr)
        
    def fire_event(self, event_name):
        self.last_event_name = event_name
        for func, input_args, output_args in self.event_handlers[event_name]:
            args = []
            for arg in input_args:
                if arg == 'engine':
                    args.append(self)
                elif hasattr(self, arg):
                    args.append(getattr(self, arg))
                else:
                    args.append(None)
            try:
                ret = func(*args)
            except FilterEventException:
                pass
            else:
                if len(output_args) > 1:
                    for attr, val in zip(output_args, ret):
                        setattr(self, attr, val)
                elif len(output_args) == 1:
                    setattr(self, output_args[0], ret)
                
    def on(self, event_name, input_args=(), output_args=()):
        
        def decorator(f):
            self.add_event_handler(event_name, f, input_args, output_args)
            return f

        return decorator

    def run(self, state):
        self.state = state
        try:
            self.fire_event(Events.STARTED)
            # while loop over epochs
            while True:
                self.fire_event(Events.EPOCH_STARTED)
                # while loop over iterations
                try:
                    while True:
                        self.batch = None
                        self.fire_event(Events.GET_BATCH_STARTED)
                        self.fire_event(Events.GET_BATCH)
                        self.fire_event(Events.GET_BATCH_COMPLETED)
                        self.fire_event(Events.ITERATION_STARTED)
                        self.fire_event(Events.PROCESS)
                        self.fire_event(Events.ITERATION_COMPLETED)
                        self.fire_event(Events.CHECKPOINT)
                except TerminateEpochException:
                    self.fire_event(Events.EPOCH_COMPLETED)
        except TerminateRunException:
            self.fire_event(Events.TERMINATE)
        self.fire_event(Events.COMPLETED)
        self.state = None
