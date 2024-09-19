# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import builtins
import dis
import sys
from itertools import chain
from typing import TYPE_CHECKING

from ...utils import InnerError, NameGenerator
from .guard import StringifiedExpression, stringify_pyobject, union_free_vars

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .pycode_generator import PyCodeGen
    from .variables import VariableBase


class Tracker:
    """
    Tracker is a base class responsible for tracking variables or objects in Python code.
    It is used to identify how a variable is derived from the initial state of the frame.

    Args:
        inputs: The list of variables to be tracked.

    Note:
        It serves as an abstract class and should not be instantiated directly.
    """

    inputs: Sequence[VariableBase]
    name_generator = NameGenerator("tracker_")

    def __init__(self, inputs: Sequence[VariableBase], changed: bool = False):
        self.inputs = inputs
        self.changed = changed
        self.id = Tracker.name_generator.next()

    def gen_instructions(self, codegen: PyCodeGen) -> None:
        """
        Generate instructions based on the tracked variables.

        Args:
            codegen (PyCodeGen): An instance of PyCodeGen to generate instructions.
        """
        raise NotImplementedError

    # TODO(xiongkun): trace_value_from_frame is not a good name, it should be more related to guard but not tracable.
    def trace_value_from_frame(self) -> StringifiedExpression:
        """
        Trace the value of the tracked variables from the frame. It used for generating the guard.

        Returns:
            The value of the tracked variables.
        """
        raise NotImplementedError

    def is_traceable(self) -> bool:
        """
        Determine if all the tracked variables can be traced from the frame.

        Returns:
            bool: True if all tracked variables are traceable, False otherwise.
        """
        if self.changed:
            return False
        for input in self.inputs:
            if not input.tracker.is_traceable():
                return False
        return True

    def need_guard(self) -> bool:
        return self.is_traceable()


class DummyTracker(Tracker):
    """
    DummyTracker is a subclass of Tracker that specifically tracks variables cannot be reproduced from the frame.
    It is mostly generated by complex operations (instructions).

    Args:
        inputs (list[VariableBase]): The input variables associated with the generated variables.
    """

    def __init__(self, inputs: Sequence[VariableBase]):
        super().__init__(inputs)

    def gen_instructions(self, codegen: PyCodeGen):
        raise InnerError("DummyTracker has no instructions")

    def trace_value_from_frame(self):
        raise InnerError("DummyTracker can't trace value from frame")

    def is_traceable(self):
        return False

    def __repr__(self) -> str:
        return f"DummyTracker(num_inputs={len(self.inputs)})"

    def need_guard(self) -> bool:
        return False


class SymbolicOperationTracker(Tracker):
    """
    SymbolicOperationTracker is a subclass of Tracker that specifically tracks variables cannot be reproduced from the frame.
    It is mostly generated by complex operations of symbolic variables.

    Args:
        inputs (list[VariableBase]): The input variables associated with the generated variables.
    """

    def __init__(self, inputs: Sequence[VariableBase], method_name: str):
        super().__init__(inputs)
        self.method_name = method_name

    def gen_instructions(self, codegen: PyCodeGen):
        raise InnerError("SymbolicOperationTracker has no instructions")

    def trace_value_from_frame(self):
        raise InnerError(
            "SymbolicOperationTracker can't trace value from frame"
        )

    def __repr__(self) -> str:
        return f"SymbolicOperationTracker(num_inputs={len(self.inputs)})"

    def is_traceable(self):
        # TODO(zrr1999): to implement gen_instructions and trace_value_from_frame
        return False

    def need_guard(self) -> bool:
        # TODO(zrr1999): to implement gen_instructions and trace_value_from_frame
        return False


class DanglingTracker(Tracker):
    """
    DanglingTracker is a subclass of Tracker that specifically tracks variables that are not in the frame.
    Variables whose tracker is DanglingTracker should not be placed on the stack, except for NullVariable.
    DanglingTracker is often used in conjunction with BuiltinVariable to reuse the dispatch mechanism.

    Examples:
        >>> import operator
        >>> from sot.opcode_translator.executor.variables import BuiltinVariable, ConstantVariable
        >>> a = ConstantVariable.wrap_literal(1, None)
        >>> b = ConstantVariable.wrap_literal(2, None)
        >>> c = BuiltinVariable(operator.add, None, DanglingTracker())(a, b)
        >>> c.value
        3
    """

    def __init__(self):
        super().__init__([])

    def gen_instructions(self, codegen: PyCodeGen):
        raise InnerError("DanglingTracker has no instructions")

    def trace_value_from_frame(self):
        raise InnerError("DanglingTracker can't trace value from frame")

    def is_traceable(self):
        return False

    def __repr__(self) -> str:
        return "DanglingTracker()"


class LocalTracker(Tracker):
    """
    LocalTracker is a subclass of Tracker that specifically tracks variables from f_locals of frame.

    Args:
        name (str): The name of the variable in f_locals to be tracked.
    """

    def __init__(self, name: str):
        super().__init__([])
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen) -> None:
        codegen.gen_load_fast(self.name)

    def trace_value_from_frame(self) -> StringifiedExpression:
        return StringifiedExpression(f"frame.f_locals['{self.name}']", [], {})

    def __repr__(self) -> str:
        return f"LocalTracker(name={self.name})"


class CellTracker(LocalTracker):
    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_deref(self.name)

    def trace_value_from_frame(self):
        return StringifiedExpression(f"frame.f_locals['{self.name}']", [], {})

    def __repr__(self) -> str:
        return f"CellTracker(name={self.name})"


class GlobalTracker(Tracker):
    """
    GlobalTracker is a subclass of Tracker that specifically tracks variables from f_globals of frame.

    Args:
        name (str): The name of the variable in f_globals to be tracked.
    """

    def __init__(self, name: str):
        super().__init__([])
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen) -> None:
        codegen.gen_load_global(self.name, push_null=False)

    def trace_value_from_frame(self) -> StringifiedExpression:
        return StringifiedExpression(f"frame.f_globals['{self.name}']", [], {})

    def __repr__(self) -> str:
        return f"GlobalTracker(name={self.name})"


class BuiltinTracker(Tracker):
    """
    BuiltinTracker is a subclass of Tracker that specifically tracks variables from f_builtins of frame.

    Args:
        name (str): The name of the variable in f_builtins to be tracked.
    """

    def __init__(self, name: str):
        super().__init__([])
        self.name = name

    def gen_instructions(self, codegen: PyCodeGen) -> None:
        codegen.gen_load_global(self.name, push_null=False)

    def trace_value_from_frame(self) -> StringifiedExpression:
        return StringifiedExpression(
            f"builtins.__dict__['{self.name}']", [], {"builtins": builtins}
        )

    def __repr__(self) -> str:
        return f"BuiltinTracker(name={self.name})"


class ConstTracker(Tracker):
    """
    ConstTracker is a subclass of Tracker that specifically tracks a constant value.

    Args:
        value (Any): The value of the constant.
    """

    def __init__(self, value):
        super().__init__([])
        self.value = value

    def gen_instructions(self, codegen: PyCodeGen):
        codegen.gen_load_const(self.value)

    def trace_value_from_frame(self):
        value_str, value_free_vars = stringify_pyobject(self.value)
        return StringifiedExpression(
            value_str, [], union_free_vars(value_free_vars)
        )

    def __repr__(self) -> str:
        return f"ConstTracker(value={self.value})"

    def need_guard(self) -> bool:
        return False


class BinaryOperatorTracker(Tracker):
    def __init__(
        self, operator: str, operands: list[VariableBase], addition=None
    ):
        """
        addition is for the case that the operator is "COMPARE_OP", which represents the dis.cmp_op's index.
        """
        super().__init__(operands, False)
        assert len(operands) == 2, "Currently only support binary operator."
        self.operands = operands
        self.operator = operator
        self.addition = addition

    def gen_instructions(self, codegen: PyCodeGen):
        for operand in self.operands:
            operand.tracker.gen_instructions(codegen)
        self.gen_operator_instr(codegen)

    def gen_operator_instr(self, codegen: PyCodeGen):
        if self.operator == "COMPARE_OP":
            codegen.gen_compare(self.addition)
        else:
            codegen.gen_operator(self.operator)

    def get_operator_symbol(self):
        if self.operator == "COMPARE_OP":
            return dis.cmp_op[self.addition]
        return {
            "BINARY_ADD": "+",
            "BINARY_SUBTRACT": "-",
            "BINARY_MUL": "*",
            "BINARY_POWER": "**",
        }[self.operator]

    def trace_value_from_frame(self):
        sub_exprs = [x.tracker.trace_value_from_frame() for x in self.operands]
        sub_frees = [x.free_vars for x in sub_exprs]
        expr = f"({{}} {self.get_operator_symbol()} {{}})"
        return StringifiedExpression(
            expr,
            list(sub_exprs),
            union_free_vars(*list(sub_frees)),
        )

    def __repr__(self) -> str:
        return f"BinaryOperatorTracker(operator={self.operator})"


class GetAttrTracker(Tracker):
    """
    GetAttrTracker is a subclass of Tracker that specifically tracks the attribute access of an variable.

    Args:
        obj (VariableBase): The object whose attribute is to be tracked.
        attr (str): The attribute to be tracked.
    """

    def __init__(self, obj: VariableBase, attr: str, changed: bool = False):
        super().__init__([obj], changed)
        self.obj = obj
        self.attr = attr

    def gen_instructions(self, codegen: PyCodeGen):
        self.obj.tracker.gen_instructions(codegen)
        codegen.gen_load_attr(self.attr)

    def trace_value_from_frame(self):
        obj_tracer = self.obj.tracker.trace_value_from_frame()
        if self.attr.isidentifier():
            expr = f"{{}}.{self.attr}"
        else:
            expr = f"getattr({{}}, '{self.attr}')"
        return StringifiedExpression(
            expr,
            [obj_tracer],
            union_free_vars(obj_tracer.free_vars),
        )

    def __repr__(self) -> str:
        return f"GetAttrTracker(attr={self.attr})"

    def need_guard(self) -> bool:
        return self.is_traceable() and self.obj.tracker.need_guard()


class GetItemTracker(Tracker):
    """
    GetItemTracker is a subclass of Tracker that specifically tracks item access of a container variable.

    It generates instructions and traces the item value from the frame.

    Args:
        container_var (VariableBase): The container object whose item is to be tracked.
        key: The key/index of the item to be tracked.
    """

    def __init__(self, container_var: VariableBase, key: object, changed=False):
        super().__init__([container_var], changed)
        self.container = container_var
        self.key = key

    def gen_instructions(self, codegen: PyCodeGen):
        self.container.tracker.gen_instructions(codegen)
        if isinstance(self.key, slice):
            codegen.gen_load_const(self.key.start)
            codegen.gen_load_const(self.key.stop)
            codegen.gen_load_const(self.key.step)
            codegen.gen_build_slice(3)
        else:
            codegen.gen_load_const(self.key)
        codegen.gen_subscribe()

    def trace_value_from_frame(self):
        container_tracer = self.container.tracker.trace_value_from_frame()
        key_string, key_free_vars = stringify_pyobject(self.key)
        return StringifiedExpression(
            f"{{}}[{key_string}]",
            [container_tracer],
            union_free_vars(container_tracer.free_vars, key_free_vars),
        )

    def __repr__(self) -> str:
        return f"GetItemTracker(key={self.key!r})"

    def need_guard(self) -> bool:
        return self.is_traceable() and self.container.tracker.need_guard()


class GetIterTracker(Tracker):
    """
    GetIterTracker is a subclass of Tracker that specifically tracks iteration of a variable.

    It generates instructions and traces the iterator from the frame.

    Args:
        iter_source (VariableBase): The source variable to be iterated.
    """

    def __init__(self, iter_source: VariableBase):
        super().__init__([iter_source])
        self.iter_source = iter_source

    def gen_instructions(self, codegen: PyCodeGen):
        self.iter_source.tracker.gen_instructions(codegen)
        codegen.add_instr("GET_ITER")

    def trace_value_from_frame(self):
        iter_source_tracer = self.iter_source.tracker.trace_value_from_frame()
        return StringifiedExpression(
            "iter({})",
            [iter_source_tracer],
            union_free_vars(iter_source_tracer.free_vars),
        )

    def __repr__(self) -> str:
        return "GetIterTracker()"


class CreateLayerTracker(Tracker):
    def __init__(self, layer_class, args, kwargs):
        super().__init__([layer_class] + list(args) + list(kwargs.values()))
        self.layer_class = layer_class
        self.args = args
        self.kwargs = kwargs

    def gen_instructions(self, codegen: PyCodeGen):
        if sys.version_info >= (3, 11):
            codegen.gen_push_null()

        self.layer_class.reconstruct(codegen)
        for variable in self.args:
            variable.reconstruct(codegen)

        if len(self.kwargs) == 0:
            codegen.gen_call_function(argc=len(self.args))
        else:
            codegen.gen_build_tuple(len(self.args))
            for k, v in self.kwargs.items():
                codegen.gen_load_const(k)
                v.reconstruct(codegen)
            codegen.gen_build_map(len(self.kwargs))
            codegen.gen_call_function_ex(has_kwargs=True)

    def trace_value_from_frame(self):
        class_tracer = self.layer_class.tracker.trace_value_from_frame()
        arg_tracers = [
            arg.tracker.trace_value_from_frame() for arg in self.args
        ]
        kwarg_tracers_dict = {
            k: v.tracker.trace_value_from_frame()
            for k, v in self.kwargs.items()
        }
        kwarg_tracers = list(kwarg_tracers_dict.values())

        expr = "{}("
        expr += ", ".join(["{}"] * len(arg_tracers))
        if len(arg_tracers) and len(kwarg_tracers) > 0:
            expr += ", "
        expr += ", ".join(f"{k}={{}}" for k in kwarg_tracers_dict.keys())
        expr += ")"

        return StringifiedExpression(
            expr,
            [class_tracer] + arg_tracers + kwarg_tracers,
            union_free_vars(
                *(
                    tracer.free_vars
                    for tracer in chain(
                        [class_tracer], arg_tracers, kwarg_tracers
                    )
                )
            ),
        )

    def __repr__(self) -> str:
        return f"CreateLayerTracker(Layer={self.layer_class}, args={self.args}, kwargs={self.kwargs})"
