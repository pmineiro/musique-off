class ParamsType(type):
    default_values = { 
            'alpha': 1000.0,
            'attn_impl': 'flash',
            'base_model_id': 'microsoft/Phi-3-medium-128k-instruct',
            'base_port': 4567,
            'clip': 1.0,
            'dataset_seed': 42,
            'distribution_matching': True,
            'do_learning': True,
            'dpo_beta': 1.0,
            'empty_cache_every': False,
            'empty_cache_per_episode': True,
            'force_extractive': False,
            'lora_r': 8,
            'max_queries': 4,
            'micro_batch_size': 4,
            'num_workers_per_gpu': 2,
            'prediction_file': '/dev/null',
            'ref_update_weight': 0.5,
            'save_every': 100,
            'seekto': 0,
            'sync_each_batch': True,
            'split': 'train',
            'tick_timeout_seconds': 0.1,
            'tick_timeout_max': 10,
            'train_on_dev': False,
            'trace': False,
            'trust_remote_code': True,
    }

    @property
    def attn_implementation(cls):
        attn_impl = cls.attn_impl
        return "flash_attention_2" if attn_impl == 'flash' else attn_impl

    @property
    def final_model_id(cls):
        import os
        return os.environ.get('final_model_id', None)

    @property
    def output_dir(cls):
        # https://amulet-docs.azurewebsites.net/main/basics/22_outputs.html#amulet-workarounds
        import os
        return os.environ.get('AMLT_DIRSYNC_DIR', '.')

    @property
    def ref_metric(cls):
        import os
        return os.environ.get('ref_metric', None)

    @property
    def sdp_kernel_args(cls):
        attn_impl = cls.attn_impl
        return { 'enable_flash': attn_impl == 'flash' or attn_impl == 'sdpa',
                 'enable_math': attn_impl == 'sdpa',
                 'enable_mem_efficient': attn_impl == 'sdpa',
               }

    @property
    def seekto(cls):
        import os
        import re
        final_model_id = cls.final_model_id
        split = cls.split
        regex = r'_(\d+)$'
        if os.environ.get('seekto', None) is not None or split != 'train' or final_model_id is None or not re.search(regex, final_model_id):
            return cls.get_generic_value('seekto')
        else:
            return int(re.search(regex, final_model_id).group(1))

    def get_generic_model_id(cls, name):
        import os
        import re
        final_model_id = cls.final_model_id
        return None if final_model_id is None else re.sub(r'final(?=_)', re.escape(name), final_model_id)

    def cast_to(cls, proto, raw):
        if type(proto) == type(True):
            return (raw == 'True')
        else:
            return type(proto)(raw)

    def post_validate(cls, name, value):
        if name == 'do_learning':
            allowed_to_train = [ 'train', 'validation' ] if cls.train_on_dev else [ 'train' ]
            assert cls.split in allowed_to_train or not value, f'do_learning is {value} but split is {cls.split}'
        elif name == 'ref_update_weight':
            assert 0 <= value <= 1, f'ref_update_weight {value} not in [0, 1]'

        if name == 'num_workers_per_gpu':
            import torch
            if torch.cuda.device_count() < 2:
                import warnings
                warnings.warn('only 1 gpu detected so forcing num_workers_per_gpu=1')

                value = 1

        return value

    def get_generic_value(cls, name):
        import os
        import re
        final_model_id = cls.final_model_id
        short_name = name.split('_')[0]
        short_name_duplicated = len([v for v in cls.default_values.keys() if v.split('_')[0] == short_name]) > 1
        if short_name_duplicated and final_model_id and f'_{short_name}_' in final_model_id:
            import warnings
            warnings.warn(f'accessing Parameter "{name}" with duplicated short name "{short_name}", not using final_model_id')

        if final_model_id is None or f'_{short_name}_' not in final_model_id or short_name_duplicated:
            raw = os.environ.get(name, str(cls.default_values[name]))
        else:
            esc_short_name = re.escape(short_name)
            regex = f'_{esc_short_name}_' + r'([^_]+)'
            try:
                raw = re.search(regex, final_model_id).group(1)
            except AttributeError:
                raise ValueError(f'cannot determine {name} ({short_name}) from {final_model_id} using {regex}')

        return cls.post_validate(name, cls.cast_to(cls.default_values[name], raw))

    def __getattr__(cls, name):
        if name in cls.default_values:
            return cls.get_generic_value(name)
        elif name.endswith('_model_id'):
            return cls.get_generic_model_id(name.split('_model_id')[0])
        else:
            raise AttributeError(f'Params has no attribute "{name}"')

class Params(metaclass=ParamsType):
    pass
