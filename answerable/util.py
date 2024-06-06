import contextlib

class EpisodeData(object):
    def __init__(self, *, ansf1, ansem, id, nquery, predicted_answerable, predicted_answer, predicted_support_idxs, suff, suppf1, pardigest, suff_raw_score):
        super().__init__()
        self.ansem = ansem
        self.ansf1 = ansf1
        self.id = id
        self.nquery = nquery
        self.predicted_answerable = predicted_answerable
        self.predicted_answer = predicted_answer
        self.predicted_support_idxs = predicted_support_idxs
        self.suff = suff
        self.suppf1 = suppf1
        self.pardigest = pardigest
        self.suff_raw_score = suff_raw_score
        # https://leaderboard.allenai.org/musique_ans/submissions/get-started
        self.prediction = { 'id': self.id,
                            'predicted_answer': self.predicted_answer,
                            'predicted_support_idxs': self.predicted_support_idxs,
                            'predicted_answerable': self.predicted_answerable,
                            'pardigest': self.pardigest,
                            'suff_raw_score': self.suff_raw_score,
                          }

class Update(object):
    def __init__(self, *, wrapper_name, update_datum):
        super().__init__()
        self.wrapper_name = wrapper_name
        self.update_datum = update_datum

class CTSig(object):
    def __init__(self, *, tokenizer):
        import re

        super().__init__()
        try:
            messages = [ { 'role': 'system', 'content': 'wazzup' }, { 'role': 'user', 'content': 'hey' } ]
            tokenizer.apply_chat_template(messages, tokenize=False)
            supports_system = True
        except:
            supports_system = False

        self._supports_system = supports_system

    @property
    def supports_system(self):
        return self._supports_system

    def render_with_system(self, *, system, user):
        if self.supports_system:
            return [ { 'role': 'system', 'content': system }, { 'role': 'user', 'content': user } ]
        else:
            return [ { 'role': 'user', 'content': f'{system}\n\n{user}' } ]

    def __str__(self):
        return str((self.supports_system,))

def make_optimizer(model, alpha):
    # TODO (?): https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html#disable-foreach-in-the-optimizer

    params = [ p for p in model.parameters() if p.requires_grad ]

    if alpha < 0:
        import bitsandbytes as bnb
        optim = bnb.optim.Adam8bit(params, lr=-alpha)
    elif alpha < 1:
        import torch
        optim = torch.optim.Adam(params, lr=alpha)
    else:
        import parameterfree
        optim = parameterfree.COCOB(params, alpha=alpha)

    return optim, sum(p.numel() for p in params)

@contextlib.contextmanager
def set_directory(path):
    import os
    from pathlib import Path

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

def start_printer_process(*, port, metrics):
    from Net import NetServer
    from Params import Params as P
    from ProgressPrinter import ProgressPrinter

    class PrintAndSave(object):
        def __init__(self, *, progress, metrics, save_every, ref_metric):
            super().__init__()

            assert ref_metric is None or ref_metric in metrics

            self._progress = progress
            self._metrics = metrics
            self._save_every = save_every
            self._ref_metric = ref_metric
            self._best_ref_metric = None

        def get_count(self):
            return self._progress.cnt

        def addobs(self, **kwargs):
            self._progress.addobs(*[kwargs[m] for m in self._metrics])
            if self._progress.cnt >= self._save_every:
                self._progress.autoprint = False
            do_print = self._progress.cnt >= self._save_every and self._progress.cnt % self._save_every == 0
            do_update_ref = False

            if do_print:
                if self._ref_metric is not None:
                    ref_metric_index = metrics.index(self._ref_metric)
                    new_metric_since = self._progress.peek_since_last(ref_metric_index) # TODO: confidence/lower bound/etc.

                    if new_metric_since is not None:
                        if self._best_ref_metric is None or self._best_ref_metric < new_metric_since:
                            self._best_ref_metric = new_metric_since
                            do_update_ref = True

                self._progress.print()

            return (do_print, self._progress.cnt, do_update_ref)

    class FileAppend(object):
        def __init__(self, *, handle):
            self._handle = handle

        def append(self, obj):
            import json
            print(json.dumps(obj), file=self._handle, flush=True)

    with set_directory(P.output_dir):
        with ProgressPrinter(*metrics, silent=P.trace) as progress, open(P.prediction_file, 'w') as handle:
            progress.offset = P.seekto
            server = NetServer(host='127.0.0.1', port=port)
            server.add_object(obj_id='printer', obj=PrintAndSave(progress=progress, metrics=metrics, save_every=P.save_every, ref_metric=P.ref_metric))
            server.add_object(obj_id='predictions', obj=FileAppend(handle=handle))

            while server.runonce(timeout=1):
                pass

    return 0

# tries to wait around a while for enough requests to show up to form a large enough batch
class StridingDeferredProxy(object):
    def __init__(self, *, all_conn, agent_id, max_tick_count):
        from collections import defaultdict

        super().__init__()

        self._all_conn = all_conn
        self._agent_id = agent_id
        self._max_tick_count = max_tick_count
        self._deferred = defaultdict(list)
        self._tick_counts = defaultdict(int)

    def _defer(self, *, op, op_args, op_kwargs):
        from Net import NetAsyncServer

        self._deferred[op].append((op_args, op_kwargs))
        assert op_args[1:] == self._deferred[op][0][0][1:], "variable additional args with a deferred operation not supported"
        assert op_kwargs == self._deferred[op][0][1], "variable kwargs with a deferred operation not supported"
        return NetAsyncServer.Deferred()

    def _strided_execute(self, *, op, op_args, op_kwargs):
        from collections.abc import Sequence
        from more_itertools import divide
        from numbers import Number

        strided_args0 = [ list(c) for c in divide(len(self._all_conn), op_args[0]) ]
        max_size = max(len(c) for c in strided_args0)
        padded_strided_args0 = [ c[:max_size] + [None]*(max_size-len(c)) for c in strided_args0 ]

        for args0, conn in zip(padded_strided_args0, self._all_conn):
            conn.send((self._agent_id, op, (args0,) + op_args[1:], op_kwargs))

        all_results = [ conn.recv() for conn in self._all_conn ]

        if isinstance(all_results[0], Number):
           # NB: use scalar aggregation (averaging)
           numerator = 0
           denominator = 0
           for args0, res in zip(padded_strided_args0, all_results):
               num_elements = sum(int(arg is not None) for arg in args0)
               if num_elements:
                   numerator += num_elements * res
                   denominator += num_elements

           return numerator / max(1, denominator)
        elif isinstance(all_results[0], Sequence) and not isinstance(all_results[0], str):
            # NB: use list aggregation
            combo_results = []
            for args0, res in zip(padded_strided_args0, all_results):
                combo_results.extend([ result for arg, result in zip(args0, res) if arg is not None ])

            assert len(combo_results) == len(op_args[0])
            return combo_results
        else:
            assert False, "should not be reached"

    def dispatch(self, op, *args, **kwargs):
        # https://docs.python.org/3/tutorial/controlflow.html#arbitrary-argument-lists
        # These arguments will be wrapped up in a tuple (see Tuples and Sequences).
        if args:
            from collections.abc import Sequence
            from numbers import Number

            if isinstance(args[0], Sequence) and not isinstance(args[0], str):
                if len(args[0]) >= len(self._all_conn):
                    # immediate execution
                    return self._strided_execute(op=op, op_args=args, op_kwargs=kwargs)
                else:
                    return self._defer(op=op, op_args=args, op_kwargs=kwargs)

        # if we get here, just use broadcasting
        for conn in self._all_conn:
            conn.send((self._agent_id, op, args, kwargs))

        for conn in self._all_conn:
            res = conn.recv()

        return res

    def tick(self):
        from collections.abc import Sequence
        from more_itertools import split_into
        from numbers import Number

        for op, pending in list(self._deferred.items()):
            total_op_args = sum(len(op_args[0]) for op_args, op_kwargs in pending)
            if self._tick_counts[op] >= self._max_tick_count or total_op_args >= len(self._all_conn):
                first_args1plus = pending[0][0][1:]
                first_kwargs = pending[0][1]
                all_op_args0 = [ arg for op_args, op_kwargs in pending for arg in op_args[0] ]
                all_op_args = (all_op_args0,) + first_args1plus
                try:
                    result = self._strided_execute(op=op, op_args=all_op_args, op_kwargs=first_kwargs)

                    if isinstance(result, Number): # just broadcast the (averaged) scalar to everybody, close enough
                        yield from (('result', op, result) for _ in pending)
                    elif isinstance(result, Sequence) and not isinstance(result, str): # distibute result to the original pending requests
                        per_pending_lens = [ len(op_args[0]) for op_args, op_kwargs in pending ]
                        per_pending_results = list(split_into(result, per_pending_lens))

                        yield from (('result', op, r) for r in per_pending_results)
                    else:
                        assert False, "should not be reached"
                except Exception as e:
                    yield from (('exception', op, f'exception {e} handling deferred execution for {op}') for _ in pending)

                del self._tick_counts[op]
                del self._deferred[op]
            else:
                self._tick_counts[op] += 1

    def __getattr__(self, op): # NB: fragile ... no graceful error handling
        return lambda *args, **kwargs: self.dispatch(op, *args, **kwargs)

def start_proxy_process(*, world_size, adapters, port, pipes):
    from Net import NetAsyncServer
    from Params import Params as P

    assert len(pipes) == world_size

    # close fsdp side
    for p in pipes:
        p[1].close()

    all_conn = [ p[0] for p in pipes ]

    for conn in all_conn:
        msg = conn.recv()
        assert msg == 'ready'
        conn.send('go')

    server = NetAsyncServer(host='127.0.0.1', port=port)

    for adapter in adapters:
        server.add_object(obj_id=adapter.suffix, obj=StridingDeferredProxy(all_conn=all_conn, agent_id=adapter.suffix, max_tick_count=P.tick_timeout_max))

    while server.runonce(timeout=P.tick_timeout_seconds):
        pass

    for conn in all_conn:
        conn.send('shutdown')

    return 0

def start_gpu_process(*, rank, world_size, adapters, pipes, dist_port):
    from datetime import timedelta
    import os
    from Params import Params as P
    #from peft import prepare_model_for_kbit_training
    from peft.utils.other import fsdp_auto_wrap_policy
    from ProgressPrinter import ProgressPrinter
    from random import Random
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, BackwardPrefetch, ShardingStrategy
    from torch import inf
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from util import make_optimizer, set_directory

    assert 0 <= rank < world_size
    assert len(pipes) == world_size

    conn = pipes[rank]
    # close proxy side
    conn[0].close()
    conn = conn[1]

    # close all other pipes
    for n, p in enumerate(pipes):
        if n != rank:
            p[0].close()
            p[1].close()

    torch.manual_seed(42)
    torch.cuda.set_device(rank)

    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(dist_port)
    dist.init_process_group(rank=rank, world_size=world_size, timeout=timedelta(seconds=1200))

    try:
        # https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora#training
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                 bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_quant_storage=torch.bfloat16,
                                                )
        # TODO: this loads the entire (quantized) model on the GPU and then FSDP redistributes it
        #       for really large models this won't fit
        #       but if i set device_map="cpu" here, it doesn't work
        #       possible fix is empty_model context manager ...

        # https://github.com/Dao-AILab/flash-attention/issues/231
        frpre_kwargs = { 'trust_remote_code': True } if P.trust_remote_code else {}
        model = AutoModelForCausalLM.from_pretrained(P.base_model_id,
                                                     quantization_config=quantization_config,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation=P.attn_implementation,
                                                     **frpre_kwargs)

        # begin peft.prepare_model_for_kbit_training ... it doesn't handle torch.bfloat16 properly, so put the important stuff inline.
        for name, param in model.named_parameters():
            # freeze base model's layers
            param.requires_grad = False
        model.gradient_checkpointing_enable() # use_reentrant=False (?)
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # end peft.prepare_model_for_kbit_training

        frpre_kwargs = { 'trust_remote_code': True } if P.trust_remote_code else {}
        tokenizer = AutoTokenizer.from_pretrained(P.base_model_id, add_special_tokens=False, **frpre_kwargs)
        assert tokenizer.eos_token is not None
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # load and expose all adapters to FSDP
        for adapter in adapters:
            adapter.load_pretrained(model=model)
        model.enable_adapters()
        model.set_adapter([ n for adapter in adapters for n in adapter.get_all_names() ])

        # check attn_implementation
        dist.barrier()
        with torch.backends.cuda.sdp_kernel(**P.sdp_kernel_args):
            model(input_ids=tokenizer([ "hi" ], return_tensors='pt', padding=False)['input_ids'], attention_mask=None)

        # https://huggingface.co/docs/peft/en/accelerate/fsdp#the-important-parts
        # Here, one main thing to note currently when using FSDP with PEFT is that use_orig_params needs to be False to realize GPU memory savings.
        fsdp_kwargs = { 'device_id': rank,
                        'auto_wrap_policy': fsdp_auto_wrap_policy(model),
                        'sharding_strategy': ShardingStrategy.FULL_SHARD,
                        'backward_prefetch': BackwardPrefetch.BACKWARD_POST,
                        'use_orig_params': False,
                      }
        model = FSDP(model, **fsdp_kwargs)
        clip_lam = lambda: model.clip_grad_norm_(P.clip, norm_type=inf)
        optimizer, pcnt = make_optimizer(model, P.alpha)

        wrappers = { adapter.suffix: adapter.init_wrapper(model=model, tokenizer=tokenizer, optimizer=optimizer, clip=clip_lam) for adapter in adapters }

        conn.send('ready')
        msg = conn.recv()
        assert msg == 'go'

        while True:
            msg = conn.recv()
            if msg == 'shutdown':
                break

            (agent_id, op, args, kwargs) = msg
            with set_directory(P.output_dir):
                result = getattr(wrappers[agent_id], op)(*args, **kwargs)
            conn.send(result)

    finally:
        dist.destroy_process_group()

def convert_to_extractive(*, guesses, datum, used):
    from collections import defaultdict
    from Musique import MetricsAnswer

    paragraphs = datum['paragraphs']

    inverse_normalized_guesses = { MetricsAnswer.normalize_answer(s) : s for s in guesses }
    guess_scores = { orig: cnt + tiebreak for normalized, orig in inverse_normalized_guesses.items()
                                          for cnt in (sum(MetricsAnswer.normalize_answer(g) == normalized for g in guesses),)
                                          for tiebreak in (0.1 if orig == guesses[0] else 0,) }

    def span_score(span):
        return len(MetricsAnswer.normalize_answer(' '.join(span)).split())

    for guess, score in sorted(guess_scores.items(), key=lambda v:v[1], reverse=True):
        words = guess.split()
        spans = [ words[n:m+1] for n in range(len(words)) for m in range(len(words)) if m >= n ]
        candidates = [ span
                       for span in spans
                       if any(' '.join(span) in p['paragraph_text'] for index in used for p in (paragraphs[index],))
                       if span_score(span) > 0 ]
        if candidates:
            adjusted_final_guess = ' '.join(max(candidates, key=span_score))
            break
    else:
        adjusted_final_guess = None

    return adjusted_final_guess
