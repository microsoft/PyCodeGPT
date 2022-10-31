
#%%
import os
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from copy import deepcopy

import ast
import unittest

import sys

import json
sys.path.append("../")

"""
Detect if a python file is a test file
"""

# print(UT_ASSERT_FUNCS)

class FuncOrVarType(int, Enum):
    constant = 0
    variable = 1
    call = 2
    package = 3
    args = 4

    def __str__(self) -> str:
        return self.name

@dataclass
class ReferenceItem:
    value_type: FuncOrVarType
    name: str
    context: List[str]
    is_stored: bool = False

    @property
    def id(self):
        return self.name

    def __str__(self) -> str:
        return str(self.value_type) + "::" + self.id

    def __repr__(self) -> str:
        return str(self)

    def is_local(self):
        return self.is_stored or self.value_type == FuncOrVarType.args

PREDEDEFINED_SORTED_FIELDS = {
    ast.For: [
        'iter', 'target', 'body'
    ],

    ast.ListComp: [
        'generators', 'elt'
    ],
}

Node2Types = {
    "condition": [
        "If"
    ],
    "loop": [
        "For",
        "While",
        "ListComp",
        "DictComp"
    ],
    "call": [
        "Call"
    ],
}

@dataclass
class FunctionModule:
    level: int
    class_name: str
    ast_node: ast.FunctionDef
    func_name: str
    references: List[ReferenceItem]
    node_stats: Dict[str, int]

    def get_short_name(self) -> str:
        return self.func_name.split('.')[-1]

def is_function_node(node: ast.AST):
    return isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)

class PycodeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()

        self.level = 0

        self.class_names = []
        self.references: List[ReferenceItem] = []
        self.call_ctx = []

        self.blocks = []

        self.node_statistics = defaultdict(int)

        self._node_positions = []

        self._func_modules = []

    def visit(self, node: ast.AST):
        """Visit a node."""
        self.level += 1
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)

        prev_ref_idx = len(self.references)
        prev_node_stats = deepcopy(self.node_statistics)

        self.node_statistics[node.__class__.__name__] += 1

        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            self._node_positions.append((node.lineno, node.end_lineno))

        visitor(node)

        self.level -= 1

        if self.level == 1 or is_function_node(node) or isinstance(node, ast.ClassDef):
            node_name = node.name if hasattr(node, "name") else str(len(self.blocks))

            block_references = self.references[prev_ref_idx:]
            node_stats = { key : self.node_statistics[key] - prev_node_stats[key] for key in self.node_statistics }

            if is_function_node(node):
                func_module = FunctionModule(
                    level=self.level,
                    ast_node=node,
                    func_name=node_name,
                    class_name=".".join(self.class_names),
                    references=block_references,
                    node_stats=node_stats
                )

                self._func_modules.append(func_module)

    def get_functions(self) -> List[FunctionModule]:
        return self._func_modules

    def add_reference(self, ref_item: ReferenceItem):
        self.references.append(ref_item)

    def generic_visit(self, node: ast.AST):
        """Called if no explicit visitor function exists for a node."""
        node_name = node.__class__.__name__
        field_names = PREDEDEFINED_SORTED_FIELDS[node.__class__] if node.__class__ in PREDEDEFINED_SORTED_FIELDS else node._fields

        for field in field_names:
            value = getattr(node, field)
            self.call_ctx.append("{}_{}".format(node_name, field))
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

            self.call_ctx.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_names.append(node.name)
        self.generic_visit(node)
        self.class_names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        is_call = len(self.call_ctx) > 0 and self.call_ctx[-1] in ["{}_func".format(ast.Call.__name__)]
        ref_item = ReferenceItem(
            value_type=FuncOrVarType.variable if not is_call else FuncOrVarType.call,
            name=node.id,
            context=[x for x in self.call_ctx],
            is_stored=isinstance(node.ctx, ast.Store) or isinstance(node.ctx, ast.AugStore)
        )

        self.add_reference(ref_item)

    def visit_Constant(self, node: ast.Constant):
        ref_item = ReferenceItem(
            value_type=FuncOrVarType.constant,
            name=str(node.value),
            context=[x for x in self.call_ctx],
            is_stored=True
        )

        self.add_reference(ref_item)

    def visit_arg(self, node: ast.arg):
        ref_item = ReferenceItem(
            value_type=FuncOrVarType.args,
            name=node.arg,
            context=[x for x in self.call_ctx],
            is_stored=True
        )

        self.add_reference(ref_item)

        self.generic_visit(node)

    def visit_alias(self, node: ast.alias):
        ref_item = ReferenceItem(
            value_type=FuncOrVarType.package,
            name=node.asname if node.asname else node.name,
            context=[x for x in self.call_ctx],
            is_stored=False
        )

        self.add_reference(ref_item)

    def visit_Attribute(self, node: ast.Attribute):
        ref_idx = len(self.references)

        self.call_ctx.append("{}_value".format(node.__class__.__name__))
        self.visit(node.value)
        self.call_ctx.pop()

        attr_values = [x.name for x in self.references[ref_idx:]]
        attr_name = ".".join(attr_values + [node.attr])

        if len(self.references) > ref_idx:
            # Pop previous ones, use longest
            self.references = self.references[:ref_idx]

            ref_item = ReferenceItem(
                value_type=FuncOrVarType.call,
                name=attr_name,
                context=[x for x in self.call_ctx],
                is_stored=isinstance(node.ctx, ast.Store) or isinstance(node.ctx, ast.AugStore)
            )

            self.add_reference(ref_item)


UT_ASSERT_FUNCS = set([key for key in unittest.TestCase.__dict__ if key.startswith("assert")])

UT_ASSERT_FUNCS.add("assertAllEqual")

FUNC_SCORE_MAPPINGS = {
    'empty': 0.2,
    'default': 1.0,
    "has_doc_string": 1.2,
    "has_test_examples": 1.3,
}

def is_assert_func_name(name: str):
    return name in UT_ASSERT_FUNCS

def is_test_func(func_name: str, call_names: List[str]) -> bool:
    return "test" in func_name.lower() and any(map(is_assert_func_name, call_names))

def is_string_expr(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)

def eval_test_file_score(example: Dict):
    file_name = example['file_name'].lower()

    # the file should have "test" keyword in its path
    if not "test" in file_name:
        return 0.0

    visitor: PycodeVisitor = example['visitor']

    funcs = visitor.get_functions()
    test_func_scores = []
    for func in funcs:
        func_name = func.func_name
        if func_name.startswith("setUp"):
            continue

        call_names = [ref.name.split('.')[-1] for ref in func.references if ref.value_type == FuncOrVarType.call]

        func_score = float(is_test_func(func_name, call_names))

        test_func_scores.append(func_score)

    score = sum(test_func_scores) / len(test_func_scores) if len(test_func_scores) > 0 else 1.0
    return score

def is_empty_func(func_node: ast.FunctionDef) -> bool:
    body = func_node.body
    idx = 0
    while idx < len(body) and is_string_expr(body[idx]):
        idx += 1

    body = body[idx:]
    if len(body) == 0:
        return True

    return len(body) == 1 and isinstance(body[-1], ast.Return) and body[-1].value is None

def has_examples_in_doc_string(doc_string: str, func_name) -> bool:
    norm_doc_string = doc_string.lower()
    return (func_name.lower() + "(") in norm_doc_string and (
        ">>>" in norm_doc_string or 'example' in norm_doc_string
    )

def get_function_score(func: FunctionModule) -> float:
    func_node = func.ast_node

    if is_empty_func(func_node):
        return FUNC_SCORE_MAPPINGS["empty"]

    if len(func_node.body) >= 1 and is_string_expr(func_node.body[0]):
        doc_string: ast.Str = func_node.body[0].value
        if has_examples_in_doc_string(doc_string.value, func.get_short_name()):
            return FUNC_SCORE_MAPPINGS["has_test_examples"]

        return FUNC_SCORE_MAPPINGS["has_doc_string"]

    return FUNC_SCORE_MAPPINGS["default"]

def eval_python_by_functions(example: Dict):
    visitor: PycodeVisitor = example['visitor']
    funcs = visitor.get_functions()

    if len(funcs) == 0:
        return FUNC_SCORE_MAPPINGS['empty']
    else:
        func_scores = list(map(get_function_score, funcs))
        return sum(func_scores) / len(func_scores)

def evaluate_by_pyast_visitor(example: Dict):
    if 'ast_obj' in example:
        example['ast'] = example['ast_obj']

    assert 'ast' in example, "Must have ast tree before evluating"
    visitor = PycodeVisitor()
    visitor.visit(example['ast'])

    example['visitor'] = visitor

    ut_score = eval_test_file_score(example)
    func_score = eval_python_by_functions(example)

    return {
        'ut_score': ut_score,
        'func_score': func_score
    }