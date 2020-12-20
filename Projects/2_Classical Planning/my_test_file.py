
import textwrap
import unittest

from itertools import chain, combinations

from aimacode.utils import expr
from aimacode.planning import Action
from aimacode.search import Node
from example_have_cake import have_cake
from air_cargo_problems import (
    air_cargo_p1, air_cargo_p2, air_cargo_p3, air_cargo_p4
)
from my_planning_graph import PlanningGraph, LiteralLayer, ActionLayer
from layers import makeNoOp, make_node


def chain_dedent(str, *args, **kwargs):
    return textwrap.dedent(str.format(*args, **kwargs)).replace("\n", " ")


class BaseMutexTest(unittest.TestCase):
    def setUp(self):
        self.cake_problem = have_cake()
        self.cake_pg = PlanningGraph(self.cake_problem, self.cake_problem.initial, serialize=False).fill()

        self.eat_action, self.bake_action = [a for a in self.cake_pg._actionNodes if not a.no_op]
        no_ops = [a for a in self.cake_pg._actionNodes if a.no_op]
        self.null_action = make_node(Action(expr('Null()'), [set(), set()], [set(), set()]))

        # some independent nodes for testing mutexes
        at_here = expr('At(here)')
        at_there = expr('At(there)')
        self.pos_literals = [at_here, at_there]
        self.neg_literals = [~x for x in self.pos_literals]
        self.literal_layer = LiteralLayer(self.pos_literals + self.neg_literals, ActionLayer())
        self.literal_layer.update_mutexes()
        
        # independent actions for testing mutex
        self.actions = [
            make_node(Action(expr('Go(here)'), [set(), set()], [set([at_here]), set()])),
            make_node(Action(expr('Go(there)'), [set(), set()], [set([at_there]), set()]))
        ]
        self.no_ops = [make_node(x) for x in chain(*(makeNoOp(l) for l in self.pos_literals))]
        self.action_layer = ActionLayer(self.no_ops + self.actions, self.literal_layer)
        self.action_layer.update_mutexes()
        for action in self.no_ops + self.actions:
            self.action_layer.add_inbound_edges(action, action.preconditions)
            self.action_layer.add_outbound_edges(action, action.effects)


class Test_1_InconsistentEffectsMutex(BaseMutexTest):
    def setUp(self):
        super().setUp()
        # bake has the effect Have(Cake) which is the logical negation of the effect 
        # ~Have(cake) from the persistence action ~NoOp::Have(cake)
        no_ops = [a for a in self.cake_pg._actionNodes if a.no_op]
        self.inconsistent_effects_actions = [self.bake_action, no_ops[3]]

        X, Y = expr('FakeFluent_X'), expr('FakeFluent_Y')
        self.fake_not_inconsistent_effects_actions = [
            make_node(Action(expr('FakeAction(X)'), [set([X]), set()], [set([X]), set()])),
            make_node(Action(expr('FakeAction(Y)'), [set([Y]), set()], [set([Y]), set()])),
        ]

    def test_1a_inconsistent_effects_mutex(self):
        acts = [self.actions[0], self.no_ops[0]]
        self.assertFalse(self.action_layer._inconsistent_effects(*acts), chain_dedent("""
            '{!s}' and '{!s}' should NOT be mutually exclusive by inconsistent effects.
            No pair of effects from {!s} and {!s} are logical opposites.
        """, acts[0], acts[1], list(acts[0].effects), list(acts[1].effects)))

    def test_1b_inconsistent_effects_mutex(self):
        acts = [self.bake_action, self.null_action]
        self.assertFalse(self.action_layer._inconsistent_effects(*acts), chain_dedent("""
            '{!s}' and '{!s}' should NOT be mutually exclusive by inconsistent effects.
            No pair of effects from {!s} and {!s} are logical opposites.
        """, acts[0], acts[1], list(acts[0].effects), list(acts[1].effects)))

    def test_1c_inconsistent_effects_mutex(self):
        acts = self.fake_not_inconsistent_effects_actions
        self.assertFalse(self.action_layer._inconsistent_effects(*acts), chain_dedent("""
            '{!s}' and '{!s}' should NOT be mutually exclusive by inconsistent effects.
            No pair of effects from {!s} and {!s} are logical opposites.
        """, acts[0], acts[1], list(acts[0].effects), list(acts[1].effects)))

    def test_1d_inconsistent_effects_mutex(self):
        acts = [self.actions[0], self.no_ops[1]]
        self.assertTrue(self.action_layer._inconsistent_effects(*acts), chain_dedent("""
            '{!s}' and '{!s}' should be mutually exclusive by inconsistent effects.
            At least one pair of effects from {!s} and {!s} are logical opposites.
        """, acts[0], acts[1], list(acts[0].effects), list(acts[1].effects)))

    def test_1e_inconsistent_effects_mutex(self):
        # inconsistent effects mutexes are static -- if they appear in any layer,
        # then they should appear in every later layer of the planning graph
        for idx, layer in enumerate(self.cake_pg.action_layers):
            if set(self.inconsistent_effects_actions) <= layer:
                self.assertTrue(layer.is_mutex(*self.inconsistent_effects_actions),
                    ("Actions {} and {} were not mutex in layer {} of the planning graph").format(
                        self.inconsistent_effects_actions[0], self.inconsistent_effects_actions[1], idx)
                )


class BaseHeuristicTest(unittest.TestCase):
    def setUp(self):
        self.cake_problem = have_cake()
        self.ac_problem_1 = air_cargo_p1()
        self.ac_problem_2 = air_cargo_p2()
        self.ac_problem_3 = air_cargo_p3()
        self.ac_problem_4 = air_cargo_p4()
        self.cake_node = Node(self.cake_problem.initial)
        self.ac_node_1 = Node(self.ac_problem_1.initial)
        self.ac_node_2 = Node(self.ac_problem_2.initial)
        self.ac_node_3 = Node(self.ac_problem_3.initial)
        self.ac_node_4 = Node(self.ac_problem_4.initial)
        self.msg = "Make sure all your mutex tests pass before troubleshooting this function."


class Test_6_MaxLevelHeuristic(BaseHeuristicTest):
    def test_6a_maxlevel(self):
        self.assertEqual(self.cake_problem.h_pg_maxlevel(self.cake_node), 1, self.msg)

    def test_6b_maxlevel(self):
        self.assertEqual(self.ac_problem_1.h_pg_maxlevel(self.ac_node_1), 2, self.msg)

    def test_6c_maxlevel(self):
        self.assertEqual(self.ac_problem_2.h_pg_maxlevel(self.ac_node_2), 2, self.msg)

    def test_6d_maxlevel(self):
        self.assertEqual(self.ac_problem_3.h_pg_maxlevel(self.ac_node_3), 3, self.msg)

    def test_6e_maxlevel(self):
        self.assertEqual(self.ac_problem_4.h_pg_maxlevel(self.ac_node_4), 3, self.msg)




if __name__ == '__main__':
    unittest.main()
