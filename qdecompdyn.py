#! /usr/bin/env python

def evaluate_conversation(*, wrappers, datum, max_queries, ctsig, do_learning, trace, micro_batch_size, force_extractive):
    import hashlib
    from more_itertools import split_into
    from Musique import MetricsAnswer, MetricsSupport
    from pprint import pformat
    from Prompts import format_for_query, format_for_ranking, format_for_stop_retrieval, format_for_intermediate, format_for_final
    from util import EpisodeData, Update, convert_to_extractive

    question = datum['question']
    paragraphs = datum['paragraphs']
    ground_truth_answers = None if datum['answer'] is None else [ datum['answer'] ] + datum['answer_aliases']
    sp_gold = None if paragraphs[0]['is_supporting'] is None else [n for n, p in enumerate(paragraphs) if p['is_supporting']]

    if do_learning:
        assert ground_truth_answers is not None
        assert sp_gold is not None

    knowledge, updates, used = [], [], []

    for nquery in range(max_queries):
        # QUERY STEP: formulate next query
        query_msg = format_for_query(question=question, knowledge=knowledge, ctsig=ctsig)
        if trace: print(f'query_msg = { pformat(query_msg) }', flush=True)
        # NB: do a rollout to compute advantages
        rawqueries = wrappers['query'].generate([query_msg], num_return_sequences=2)[0]
        if not do_learning:
            rawqueries = [ rawqueries[0] ]
        queries = [ ' '.join(q.split()) for q in rawqueries ]
        if trace: print(f'query = { queries[0] }', flush=True)

        # RETRIEVE STEP: retrieve documents
        all_score_msgs, all_orig_indices = zip(*[ format_for_ranking(question=q, knowledge=paragraphs, used=used, ctsig=ctsig) for q in queries ])
        if trace: print(f'score_msgs[0] = { pformat(all_score_msgs[0][0]) }', flush=True)
        all_scores = list(split_into(wrappers['relevance'].score([ v for s in all_score_msgs for v in s ], micro_batch_size=micro_batch_size),
                                     [ len(s) for s in all_score_msgs ]))
        all_score_choices = [ [ v[0] for v in sorted(enumerate(s), key=lambda v:v[1], reverse=True) ] for s in all_scores ]
        all_choices = [ idx[sc[0]] for sc, idx in zip(all_score_choices, all_orig_indices) ]

        if do_learning:
            # immediate supervised feedback for QUERY STEP
            ranked_queries = sorted([ (paragraphs[c]['is_supporting'], q) for c, q in zip(all_choices, queries) ], reverse=True)
            query_positive, query_negative = ranked_queries[0][1], ranked_queries[-1][1]
            query_reward_diff = ranked_queries[0][0] - ranked_queries[-1][0]
            updates.append(Update(wrapper_name='query', update_datum=(query_msg, query_positive, query_negative, query_reward_diff)))

            # immediate supervised feedback for RETRIEVE STEP
            if any(    p['is_supporting'] for n, p in enumerate(paragraphs) if n not in used) and \
               any(not p['is_supporting'] for n, p in enumerate(paragraphs) if n not in used):
                best_score_choice = next(idx for idx in all_score_choices[0]
                                             for orig_idx in (all_orig_indices[0][idx],)
                                             if paragraphs[orig_idx]['is_supporting'])
                updates.append(Update(wrapper_name='relevance', update_datum=(all_score_msgs[0][best_score_choice], 'Yes', 1)))
                second_best_score_choice = next(idx for idx in all_score_choices[0]
                                                for orig_idx in (all_orig_indices[0][idx],)
                                                if not paragraphs[orig_idx]['is_supporting'])
                updates.append(Update(wrapper_name='relevance', update_datum=(all_score_msgs[0][second_best_score_choice], 'No', 1)))

        choice = all_choices[0]
        fact = paragraphs[choice]
        knowledge.append(fact)
        used.append(choice)

        # STOP RETRIEVAL STEP: optionally stop subquestions
        if nquery + 1 < max_queries:
            from math import log

            stop_msg = format_for_stop_retrieval(question=question, knowledge=knowledge, ctsig=ctsig)
            if trace: print(f'stop_msg = { pformat(stop_msg) }', flush=True)
            stop_scores = wrappers['stop'].score([ stop_msg ], micro_batch_size=1)
            action = "Yes" if stop_scores[0] > -log(2) else "No"

            if do_learning:
               # immediate supervised feedback for STOP RETRIEVAL STEP
               should_stop = not any(p['is_supporting'] for n, p in enumerate(paragraphs) if n not in used)
               updates.append(Update(wrapper_name='stop', update_datum=(stop_msg, "Yes" if should_stop else "No", 1)))

            if action == "Yes":
                break

    # INTERMEDIATE ANSWER STEP: form intermediate answer
    intermediate_msg = format_for_intermediate(question=question, knowledge=knowledge, ctsig=ctsig)
    if trace: print(f'intermediate_msg = { pformat(intermediate_msg) }', flush=True)
    # NB: do rollouts to compute advantages
    intermediate_guesses = wrappers['intermediate'].generate([intermediate_msg], num_return_sequences=4, num_beams=16)[0]
    if not do_learning and not force_extractive:
        intermediate_guesses = [ intermediate_guesses[0] ]

    # FINAL ANSWER STEP: form final answer
    final_msgs = [ format_for_final(question=question, guess=guess, ctsig=ctsig) for guess in intermediate_guesses ]
    if trace: print(f'final_msg = { pformat(final_msgs[0]) }', flush=True)
    final_guesses = [ v[0] for v in wrappers['final'].generate(final_msgs) ]
    all_answerf1s = [ MetricsAnswer.compute_f1_max(a_golds=ground_truth_answers, a_pred=guess) for guess in final_guesses ]

    if do_learning:
        ranked_intermediates = [ v for v in sorted(zip(all_answerf1s[1:], intermediate_guesses[1:]), reverse=True) ]
        best_ranked_intermediate = ranked_intermediates[0][1]
        greedy_intermediate = intermediate_guesses[0]
        reward_diff = all_answerf1s[0] - ranked_intermediates[0][0]
        intermediate_positive, intermediate_negative = (greedy_intermediate, best_ranked_intermediate) if reward_diff >= 0 else (best_ranked_intermediate, greedy_intermediate)

        # NB: immediate reward for INTERMEDIATE ANSWER STEP
        updates.append(Update(wrapper_name='intermediate', update_datum=(intermediate_msg, intermediate_positive, intermediate_negative, abs(reward_diff))))

        # NB: immediate reward for FINAL ANSWER STEP
        updates.append(Update(wrapper_name='final', update_datum=(final_msgs[0], ground_truth_answers[0], 1)))

    final_guess = final_guesses[0]
    if force_extractive and not any(final_guess in p['paragraph_text'] for index in used for p in (paragraphs[index],)):
        adjusted_final_guess = convert_to_extractive(guesses=final_guesses, datum=datum, used=used)
        if adjusted_final_guess:
            final_guess = adjusted_final_guess
    answerf1 = MetricsAnswer.compute_f1_max(a_golds=ground_truth_answers, a_pred=final_guess)
    answerem = MetricsAnswer.compute_exact_max(a_golds=ground_truth_answers, a_pred=final_guess)
    supportf1 = MetricsSupport.supportf1(sp_pred=used, sp_gold=sp_gold)

    if trace:
        precision = sum(1 for index in used if paragraphs[index]['is_supporting']) / max(1, len(used))
        recall = sum(1 for index in used if paragraphs[index]['is_supporting'])  / max(1, sum(1 for p in paragraphs if p['is_supporting']))
        print('****************', flush=True)
        print(f'question = {question}', flush=True)
        print(f'retrieval = {pformat([ paragraphs[index]["is_supporting"] for index in used ])}', flush=True)
        print(f'precision = {precision} recall = {recall} supportf1 = {supportf1}', flush=True)
        print(f'final_guess = {final_guess} ground_truth_answers = {pformat(ground_truth_answers)} answerem = {answerem}, answerf1 = {answerf1}', flush=True)

    # NB: (id, pardigest) is unique in the full dev/test dataset files
    partext = '\n\n'.join(p['paragraph_text'] for p in paragraphs)
    pardigest = hashlib.sha256(partext.encode('utf-8')).hexdigest()
    edata = EpisodeData(ansf1 = answerf1, ansem = answerem, id = datum['id'], nquery = nquery,
                        predicted_answerable = True, predicted_answer = final_guess, predicted_support_idxs = sorted(set(used)),
                        suff = 1, suppf1 = supportf1, pardigest = pardigest, suff_raw_score = 1)

    return edata, updates

def generate_data(*, rank, world_size, wrappers, seekto, split, do_learning, max_queries, ctsig, trace, micro_batch_size, distribution_matching, force_extractive, dataset_seed):
    from datasets import load_dataset
    from random import Random
    import torch.distributed as dist

    rgen = Random(2112 + rank)

    if split == 'test':
        dataset = load_dataset('./musique_dataset.py', 'answerable', trust_remote_code=True)
    else:
        dataset_id = 'bdsaglam/musique-raw'
        dataset = load_dataset(dataset_id)

    if do_learning:
        iterable_dataset = dataset[split].to_iterable_dataset(num_shards=128)
        shuffled_dataset = iterable_dataset.shuffle(seed=dataset_seed, buffer_size=10000)
    else:
        shuffled_dataset = dataset[split]

    for n, datum in enumerate(shuffled_dataset):
        if n < seekto:
            continue

        if n % world_size != rank:
            continue

        if split == 'train' and distribution_matching:
            freqs = { 2: 0.25, 3: 0.5, 4: 1 }
            support_count = sum(1 for p in datum['paragraphs'] if p['is_supporting'])

            if rgen.random() > freqs[support_count]:
                continue

        dist.barrier()

        if split == 'test':
            # mitigate differences between musique-raw and this version
            # TODO: understand dataset library better and fix this in ./musique_dataset.py

            paragraphs = [ { 'idx': idx, 'title': title, 'paragraph_text': text, 'is_supporting': is_supporting }
                           for p in (datum['paragraphs'],)
                           for idx, title, text, is_supporting in zip(p['idx'], p['title'], p['paragraph_text'], p['is_supporting'],)
                         ]
            datum['paragraphs'] = paragraphs

        yield evaluate_conversation(wrappers = wrappers,
                                    datum = datum,
                                    max_queries = max_queries,
                                    ctsig = ctsig,
                                    do_learning = do_learning,
                                    trace = trace,
                                    micro_batch_size = micro_batch_size,
                                    force_extractive = force_extractive)

#--------------------------------------------------------------------

def start_worker_process(*, rank, num_workers, adapters, proxy_port, printer_port, dist_port, metrics, ctsig, save_pathspec_prefix):
    from datetime import timedelta
    from Net import NetClient, wait_for_server
    import os
    from Params import Params as P
    from random import Random
    import torch.distributed as dist

    rgen = Random(8675309)

    wait_for_server(host='127.0.0.1', port=proxy_port, alive_check=lambda: True)
    wrappers = { adapter.suffix: NetClient(obj_id=adapter.suffix, host='127.0.0.1', port=proxy_port) for adapter in adapters }

    wait_for_server(host='127.0.0.1', port=printer_port, alive_check=lambda: True)
    printer = NetClient(obj_id='printer', host='127.0.0.1', port=printer_port)
    predictions = NetClient(obj_id='predictions', host='127.0.0.1', port=printer_port)

    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(dist_port)
    dist.init_process_group(rank=rank, world_size=num_workers, timeout=timedelta(seconds=1200))

    try:
        if P.train_on_dev:
            dist.barrier()
            for suffix, client in wrappers.items():
                client.set_ref_to_trained(weight=1)

        for edata, updates in generate_data(rank=rank, world_size=num_workers, wrappers=wrappers, seekto=P.seekto, split=P.split,
                                            do_learning=P.do_learning, max_queries=P.max_queries, ctsig=ctsig,
                                            trace=(rank == 0 and P.trace), micro_batch_size=P.micro_batch_size,
                                            distribution_matching=P.distribution_matching, force_extractive=P.force_extractive,
                                            ataset_seed=P.dataset_seed):
            # NB: post-processing will be required b/c things can get reordered a bit and
            # https://leaderboard.allenai.org/musique_ans/submissions/get-started says
            # "Also note that the question predictions in the predictions jsonl file (i.e., id field) must be in the same order as that of the input dataset jsonl file."
            # EXTRA COMPLEXITY: id-s are not unique for the full dataset (!)
            predictions.append(edata.prediction)
            if P.empty_cache_per_episode:
                for w in wrappers.values():
                    w.empty_cache()

            results = { k: getattr(edata, k) for k in metrics if hasattr(edata, k) } | { k: 0 for k in metrics if k.endswith('up') } | { k: None for k in metrics if k.endswith('loss') }

            assert all(u.wrapper_name in wrappers for u in updates), f'bad wrapper_name: {next(u.wrapper_name for u in updates if u.wrapper_name not in wrappers)}'

            for name, adapter in sorted(wrappers.items()):
                my_updates = [ u for u in updates if u.wrapper_name == name ]
                my_update_data = [ u.update_datum for u in my_updates if u.update_datum[-1] > 0 ]
                if my_update_data:
                    rgen.shuffle(my_update_data)
                    if P.do_learning:
                        from numbers import Number
                        loss = adapter.learn(my_update_data,
                                             sync_each_batch=P.sync_each_batch,
                                             empty_cache=P.empty_cache_every,
                                             micro_batch_size=1)
                        if not isinstance(loss, Number):
                            print(f'learn threw exception for adapter {name}: {loss}', flush=True)
                            raise AssertionError(f"unexpected value for loss from adapter {name}: {loss}")

                        upname = f'{name[:3]}up'
                        results[upname] = len(my_update_data)
                        metric = f'{name[:3]}loss'
                        results[metric] = loss

            do_save, cnt, do_ref_load = printer.addobs(**results)
            if do_save and P.do_learning:
                for suffix, client in wrappers.items():
                    client.save_pretrained(pathspec = [save_pathspec_prefix, str(cnt + P.seekto)])

                if do_ref_load and P.ref_update_weight > 0:
                    for suffix, client in wrappers.items():
                        client.set_ref_to_trained(weight=P.ref_update_weight)

            if P.empty_cache_per_episode:
                for w in wrappers.values():
                    w.empty_cache()
    finally:
        dist.barrier()
        if P.do_learning:
            for suffix, client in wrappers.items():
                client.save_pretrained(pathspec = [save_pathspec_prefix, "final"])
        dist.barrier()
        # https://discuss.pytorch.org/t/when-should-i-call-dist-destory-process-group/174299/3
        # In general, destroy does not actually need to be called if the program is going to shutdown anyways.
        if rank == 0:
            from Net import send_shutdown

            send_shutdown(host='127.0.0.1', port=printer_port)
            send_shutdown(host='127.0.0.1', port=proxy_port)

def do_training(rank, world_size, num_workers, adapters, gpu_dist_port, proxy_port, pipes, printer_port, metrics, ctsig, save_pathspec_prefix, worker_dist_port):
    if rank == 0:
        from util import start_printer_process
        return start_printer_process(port=printer_port, metrics=metrics)
    if 1 <= rank < 1 + world_size:
        from util import start_gpu_process
        return start_gpu_process(rank=rank-1, world_size=world_size, adapters=adapters, pipes=pipes, dist_port=gpu_dist_port)
    elif rank == world_size + 1:
        from util import start_proxy_process
        return start_proxy_process(world_size=world_size, adapters=adapters, port=proxy_port, pipes=pipes)
    elif world_size + 2 <= rank < world_size + 2 + num_workers:
        return start_worker_process(rank=rank-(world_size+2),
                                    num_workers=num_workers,
                                    adapters=adapters,
                                    proxy_port=proxy_port,
                                    printer_port=printer_port,
                                    metrics=metrics,
                                    ctsig=ctsig,
                                    save_pathspec_prefix=save_pathspec_prefix,
                                    dist_port=worker_dist_port)

    assert False, "should not be reached"

def setup_everything(*, save_pathspec_prefix):
    from Agents import Adapter, PGAgent, DPOAgent, YesNoAgent
    import multiprocessing as mp
    from Params import Params as P
    from peft import LoraConfig, TaskType
    import torch
    from transformers import AutoTokenizer
    from util import CTSig

    assert torch.cuda.is_available()
    world_size = torch.cuda.device_count()
    assert P.num_workers_per_gpu >= 1

    frpre_kwargs = { 'trust_remote_code': True } if 'Phi-3' in P.base_model_id else {}
    tokenizer = AutoTokenizer.from_pretrained(P.base_model_id, add_special_tokens=False, **frpre_kwargs)
    assert tokenizer.eos_token is not None
    if tokenizer.pad_token is None:
        import warnings
        warnings.warn('setting pad token to eos token')
        tokenizer.pad_token = tokenizer.eos_token
    ctsig = CTSig(tokenizer=tokenizer)

    if P.do_learning:
        print(f'alpha={P.alpha} clip={P.clip} dataset_seed={P.dataset_seed} distribution_matching={P.distribution_matching} dpo_beta={P.dpo_beta} lora_r={P.lora_r} ref_metric={P.ref_metric} ref_update_weight={P.ref_update_weight} save_every={P.save_every} sync_each_batch={P.sync_each_batch}', flush=True)
    print(f'attn_impl={P.attn_impl} base_model_id={P.base_model_id} ctsig={ctsig} empty_cache={(P.empty_cache_every, P.empty_cache_per_episode)} final_model_id={P.final_model_id} force_extractive={P.force_extractive} max_queries={P.max_queries} micro_batch_size={P.micro_batch_size} num_workers_per_gpu={P.num_workers_per_gpu} prediction_file={P.prediction_file} seekto={P.seekto} split={P.split}', flush=True)

    proto_target_modules = ['qkv_proj'] if 'Phi-3' in P.base_model_id else ['q_proj', 'v_proj']
    prototype = LoraConfig(r=P.lora_r, task_type=TaskType.CAUSAL_LM, target_modules=proto_target_modules, use_rslora=True)

    adapters = [ Adapter(cls=DPOAgent, suffix='query', model_id=P.query_model_id, prototype=prototype, empty_cache=P.empty_cache_every,
                         init_kwargs={ 'beta': P.dpo_beta, 'generate_kwargs': { 'do_beam': True, 'diverse_beam': True, 'max_new_tokens': 50 } }),
                 Adapter(cls=YesNoAgent, suffix='relevance', model_id=P.relevance_model_id, prototype=prototype, empty_cache=P.empty_cache_every, init_kwargs={}),
                 Adapter(cls=YesNoAgent, suffix='stop', model_id=P.stop_model_id, prototype=prototype, empty_cache=P.empty_cache_every, init_kwargs={}),
                 Adapter(cls=DPOAgent, suffix=f'intermediate', model_id=P.intermediate_model_id, prototype=prototype, empty_cache=P.empty_cache_every,
                         init_kwargs={ 'beta': P.dpo_beta, 'generate_kwargs': { 'do_beam': True, 'diverse_beam': True, 'max_new_tokens': 100 } }),
                 Adapter(cls=PGAgent, suffix='final', model_id=P.final_model_id, prototype=prototype, empty_cache=P.empty_cache_every,
                         init_kwargs={ 'generate_kwargs': { 'do_beam': True, 'explore': False, 'max_new_tokens': 25 } }),
               ]
    metrics = [ 'ansf1', 'ansem', 'suppf1', 'nquery' ] + [ f'{adapter.suffix[:3]}{wut}' for adapter in sorted(adapters, key=lambda z:z.suffix) for wut in ('loss', 'up',) ]
    gpu_dist_port, printer_port, proxy_port, worker_dist_port = [ P.base_port + n for n in range(4) ]

    # NB: magically this is ok ...
    # https://stackoverflow.com/questions/75012853/how-do-python-pipe-still-work-through-spawning-processes
    pipes = [ mp.Pipe() for _ in range(world_size) ]

    num_workers = P.num_workers_per_gpu * world_size
    torch.multiprocessing.spawn(do_training,
                                args=(world_size, num_workers, adapters, gpu_dist_port,
                                      proxy_port, pipes, printer_port, metrics,
                                      ctsig, save_pathspec_prefix, worker_dist_port),
                                nprocs=world_size + 2 + num_workers,
                                join=True)

if __name__ == '__main__':
    # NB: the purpose of the multiprocessing complexity is to atomize the computation graph so that model parallelism avoids deadlock with dynamic graphs
    setup_everything(save_pathspec_prefix="save_musique_qdecompdyn")
