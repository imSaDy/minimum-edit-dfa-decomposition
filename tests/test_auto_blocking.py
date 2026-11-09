"""Automatic SAT blocking, exact rejection counters, and server protocol."""
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from dfa import DFA
import dfa_identify.minimum_edit as m
from dfa_identify.cli import main as cli_main
from test_minimum_edit import (
    trailing_b_mod_3, even_a_and_even_b, even_a_and_b_mod_3,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import blocking_campaign as campaign


def random_three_state_target(rng, accepting=None):
    transitions = {(q, a): rng.randrange(3) for q in range(3) for a in ("a", "b")}
    labels = [bool(rng.randrange(2)) for q in range(3)] if accepting is None else [accepting] * 3
    return DFA(start=0, states=(0,1,2), inputs={"a", "b"}, outputs={False,True},
               label=lambda q: labels[q], transition=lambda q,a: transitions[q,a])


def assert_languages(result):
    first, second = result.factors
    assert result.repaired == (first & second)
    left, right = result.witnesses
    assert first.label(left) and not second.label(left)
    assert second.label(right) and not first.label(right)
    assert result.common_witness is not None
    assert first.label(result.common_witness) and second.label(result.common_witness)
    assert result.repaired.label(result.common_witness)


def reference():
    default = {"optimization":"sat", "sat_solver":"g3", "factor_sizes":None,
               "factor_order":"max", "strictness":True, "strictness_mode":"finite",
               "product_encoding":"factored", "bfs_parent_redundancy":True}
    return {"campaign_id":"baseline", "immutable":{"options":{
        "variants":[{"id":"T01","configuration":default}, {"id":"T08","configuration":{
            **default,"bfs_parent_redundancy":False}}], "state_counts":[3,4], "workers":13,
        "timeout_seconds":28800, "repetitions":1, "memory_gib_per_test":16,
        "minimum_available_ram_gib":32}, "corpus":[
            {"input_kind":"minimal", "case_id":"sample", "states":3, "input_dot_sha256":"hash"}]}}


class AutoBlockingTests(unittest.TestCase):
    def test_real_optima_match_encoded_witnesses_and_independent_language_checks(self):
        rng = random.Random(20261014)
        targets = [trailing_b_mod_3(), even_a_and_even_b(), even_a_and_b_mod_3(),
                   random_three_state_target(rng, True), random_three_state_target(rng, False)]
        targets += [random_three_state_target(rng) for _ in range(6)]
        for target in targets:
            for ordering in ("sum", "max", "product"):
                baseline = m.find_minimum_edit_decomposition(target, optimization="sat", factor_order=ordering)
                lazy = m.find_minimum_edit_decomposition(target, optimization="sat", factor_order=ordering, strictness=False)
                self.assertEqual(lazy.edit_distance, baseline.edit_distance)
                self.assertEqual(tuple(len(f.states()) for f in lazy.factors),
                                 tuple(len(f.states()) for f in baseline.factors))
                self.assertEqual(lazy.strictness_strategy,"blocking")
                assert_languages(lazy)
        for kind in ("classic", "compact"):
            for product in ("factored", "direct"):
                lazy = m.find_minimum_edit_decomposition(even_a_and_even_b(), optimization="sat",
                    strictness=False, strictness_mode="infinite", bfs_encoding=kind, product_encoding=product)
                self.assertEqual(lazy.edit_distance,0)
                assert_languages(lazy)

    def test_unsuccessful_bounds_preserve_actual_solver_and_rejection_counts(self):
        target = random_three_state_target(random.Random(8), True)
        original_solver, original_inspect = m.Solver, m._inspect_encoded_model
        calls, rejected, models = [0], [0], [0]

        class CountingSolver:
            def __init__(self, *args, **kwargs):
                self.native=original_solver(*args, **kwargs)
            def __enter__(self):
                self.native.__enter__()
                return self
            def __exit__(self, *args):
                return self.native.__exit__(*args)
            def __getattr__(self, name):
                return getattr(self.native, name)
            def solve(self, *args, **kwargs):
                calls[0]+=1
                return self.native.solve(*args, **kwargs)

        def inspect(*args, **kwargs):
            result=original_inspect(*args, **kwargs)
            models[0]+=1
            rejected[0]+=result[1] is not None
            return result

        with patch.object(m,"Solver",CountingSolver), patch.object(m,"_inspect_encoded_model",inspect):
            result=m.find_minimum_edit_decomposition(target,optimization="sat",factor_order="max",strictness=False)
        self.assertGreater(rejected[0],0)
        self.assertEqual(result.solver_calls,calls[0])
        self.assertEqual(result.blocked_models,rejected[0])
        self.assertEqual(result.checked_models,models[0])
        self.assertTrue(any(a.edit_distance is None for a in result.factor_size_attempts))
        assert_languages(result)

    def test_cli_automatic_blocking_metadata_and_configuration(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name)/"result.json"
            # Use a JSON target to avoid assuming an example filename.
            source=Path(name)/"input.json"
            source.write_text(json.dumps({"states":[0,1,2],"alphabet":["a","b"],"start":0,"accepting":[2],
                "transitions":[{"from":q,"symbol":a,"to":(q+1)%3 if a=="b" else 0}
                               for q in range(3) for a in ("a","b")]}))
            with contextlib.redirect_stdout(io.StringIO()), \
                 patch("dfa_identify.cli.write_decomposition_visualisations",return_value={}):
                cli_main([str(source),"--no-strictness","--optimization","sat","--factor-order","max","-o",str(output)])
            payload=json.loads(output.read_text())
            self.assertEqual(payload["strictness_strategy"],"blocking")
            self.assertEqual(payload["factor_size_search"]["mode"],"all-proper-unordered")
            self.assertEqual(payload["edit_distance"],1)
            self.assertTrue(payload["factor_size_search"]["completed"])

    def test_campaign_changes_only_the_strictness_check_strategy(self):
        plan=campaign.make_plan(reference())
        self.assertEqual(plan["execution"]["variant_sequence"],["T20"])
        c=plan["variants"][0]["configuration"]
        self.assertFalse(c["strictness"])
        self.assertEqual({**c,"strictness":True},
                         {**reference()["immutable"]["options"]["variants"][0]["configuration"],"bfs_encoding":"classic"})
        self.assertEqual(plan["execution"]["timeout_seconds"],28800)

    def test_worker_adapter_selects_blocking_and_checks_result_strategy(self):
        source='''
def protocol(c):
    if c["factor_sizes"] is not None or c["strictness"] is not True:
        raise ValueError("this campaign requires automatic sizes and encoded strictness")
def execute(output, job, c):
    argv = [str(output / job["case"]["input"]), "--strictness", "--optimization", c["optimization"],
                "--factor-order", c["factor_order"], "--sat-solver", c["sat_solver"],]
    expected = {"strictness_strategy": "witness", "product_encoding": c["product_encoding"],}
    return argv, expected
'''
        adapted=campaign.adapt_harness_source(source)
        self.assertEqual(campaign.adapt_harness_source(adapted),adapted)
        namespace={}
        exec(compile(adapted,"harness.py","exec"),namespace)
        c=campaign.make_plan(reference())["variants"][0]["configuration"]
        namespace["protocol"](c)
        argv,expected=namespace["execute"](Path("output"),{"case":{"input":"source.dot"}},c)
        self.assertIn("--no-strictness",argv)
        self.assertNotIn("--strictness",argv)
        self.assertEqual(expected["strictness_strategy"],"blocking")
        c["strictness"]=True
        argv,expected=namespace["execute"](Path("output"),{"case":{"input":"source.dot"}},c)
        self.assertIn("--strictness",argv)
        self.assertEqual(expected["strictness_strategy"],"witness")


if __name__=="__main__":
    unittest.main()
