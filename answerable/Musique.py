# NB: For the full version of the dataset ...
# these metrics are only invoked if gold sufficiency is true
# https://github.com/StonyBrookNLP/musique/blob/52cd23e533bfd7439094a9e69a02139d0ba1f6ae/evaluate_v1.0.py#L58

# https://github.com/StonyBrookNLP/musique/blob/52cd23e533bfd7439094a9e69a02139d0ba1f6ae/metrics/support.py#L21
class MetricsSupport:
    @staticmethod
    def supportf1(*, sp_pred, sp_gold):
        if sp_gold is None:
            return None

        # Taken from hotpot_eval
        cur_sp_pred = set(map(int, sp_pred))
        gold_sp_pred = set(map(int, sp_gold))
        tp, fp, fn = 0, 0, 0
        for e in cur_sp_pred:
            if e in gold_sp_pred:
                tp += 1
            else:
                fp += 1
        for e in gold_sp_pred:
            if e not in cur_sp_pred:
                fn += 1
        prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        em = 1.0 if fp + fn == 0 else 0.0

        # In case everything is empty, set both f1, em to be 1.0.
        # Without this change, em gets 1 and f1 gets 0
        if not cur_sp_pred and not gold_sp_pred:
            f1, em = 1.0, 1.0
            f1, em = 1.0, 1.0

        return f1

# https://github.com/StonyBrookNLP/musique/blob/52cd23e533bfd7439094a9e69a02139d0ba1f6ae/metrics/answer.py#L1
class MetricsAnswer:
    @staticmethod
    def normalize_answer(s):
        import re
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            import string
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def get_tokens(s):
        if not s:
            return []
        return MetricsAnswer.normalize_answer(s).split()

    @staticmethod
    def _compute_exact(*, a_gold, a_pred):
        return int(MetricsAnswer.normalize_answer(a_gold) == MetricsAnswer.normalize_answer(a_pred))

    @staticmethod
    def compute_exact_max(*, a_golds, a_pred):
        if a_golds is None:
            return None

        return max(MetricsAnswer._compute_exact(a_gold=a_gold, a_pred=a_pred) for a_gold in a_golds)

    @staticmethod
    def _compute_f1(*, a_gold, a_pred):
        import collections

        gold_toks = MetricsAnswer.get_tokens(a_gold)
        pred_toks = MetricsAnswer.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def compute_f1_max(*, a_golds, a_pred):
        if a_golds is None:
            return None

        return max(MetricsAnswer._compute_f1(a_gold=a_gold, a_pred=a_pred) for a_gold in a_golds)
