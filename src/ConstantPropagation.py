from typing import List, Tuple, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ControlledGate, Gate, Qubit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode
from qiskit.quantum_info import Operator

from qcp.src.util.QubitState import QubitStateOrTop, QubitState, EPS
from qcp.src.util.ActivationState import ActivationState
from qcp.src.util.UnionTable import UnionTable

__all__ = ["ConstantPropagation"]


def _single_qubit_matrix(instr: Instruction) -> List[complex]:
    """Return a flat list [a, b, c, d] representing a 2x2 unitary."""
    base = instr.base_gate if isinstance(instr, ControlledGate) else instr
    mat = Operator(base).data
    if mat.shape != (2, 2):
        raise ValueError("Instruction is not a single-qubit unitary")
    return [complex(mat[0, 0]), complex(mat[0, 1]), complex(mat[1, 0]), complex(mat[1, 1])]


def _two_qubit_matrix(instr: Instruction) -> List[List[complex]]:
    """Return a 4x4 nested list for two-qubit instr."""
    base = instr.base_gate if isinstance(instr, ControlledGate) else instr
    mat = Operator(base).data
    if mat.shape != (4, 4):
        raise ValueError("Instruction is not a two-qubit unitary")
    return [[complex(mat[r, c]) for c in range(4)] for r in range(4)]


def _three_qubit_matrix(instr: Instruction) -> List[List[complex]]:
    """Return an 8x8 nested list for three-qubit instr."""
    base = instr.base_gate if isinstance(instr, ControlledGate) else instr
    mat = Operator(base).data
    if mat.shape != (8, 8):
        raise ValueError("Instruction is not a three-qubit unitary")
    return [[complex(mat[r, c]) for c in range(8)] for r in range(8)]


def _four_qubit_matrix(instr: Instruction) -> List[List[complex]]:
    """Return a 16x16 nested list for four-qubit instr."""
    base = instr.base_gate if isinstance(instr, ControlledGate) else instr
    mat = Operator(base).data
    if mat.shape != (16, 16):
        raise ValueError("Instruction is not a four-qubit unitary")
    return [[complex(mat[r, c]) for c in range(16)] for r in range(16)]


IGNORED_GATES: set[str] = {
    "barrier",
    "delay",
}

UNSUPPORTED_GATES: set[str] = {
    "peres",
    "peresdg",
    "atru",
    "afalse",
    "multi_atru",
    "multi_afalse",
}

RESET_NAME = "reset"
MEASURE_NAME = "measure"


class ConstantPropagation:
    MAX_AMPLITUDES: int = 1024 # и прочекать че будет


    @staticmethod
    def _check_amplitude(table: UnionTable, max_amplitudes: int, index: int) -> bool:
        reg = table[index]
        if reg.is_qubit_state() and reg.get_qubit_state().size() > max_amplitudes:
            table.set_top(index)
            return True
        return False

    @classmethod
    def _check_amplitudes(cls, table: UnionTable, max_amplitudes: int) -> None:
        for i in range(table.size()):
            cls._check_amplitude(table, max_amplitudes, i)

    @classmethod
    def _is_no_op(cls, table: UnionTable, instr: Instruction, qargs: Sequence[Qubit], max_amplitudes: int) -> bool:
        # Conservative policy: without full information (TOP), do not remove gates.
        if not qargs or any(table.is_top(q._index) for q in qargs):
            return False

        trial = table.clone()
        cls._apply_gate(trial, instr, qargs, max_amplitudes)
        return trial == table

    @classmethod
    def _propagate(
        cls,
        circuit: QuantumCircuit,
        max_amplitudes: int | None = None,
        table: UnionTable | None = None,
    ) -> Tuple[UnionTable, QuantumCircuit]:
        max_amplitudes = max_amplitudes or cls.MAX_AMPLITUDES

        table = table or UnionTable(circuit.num_qubits)
        new_circ = QuantumCircuit(*circuit.qregs, *circuit.cregs)

        for inst in circuit.data:
            instr = inst.operation
            qargs = inst.qubits
            cargs = inst.clbits

            q_indices = [q._index for q in qargs]
            name_lc = instr.name.lower()

            cls._check_amplitudes(table, max_amplitudes)

            if table.all_top() and name_lc not in (RESET_NAME, MEASURE_NAME):
                new_circ.append(instr, qargs, cargs)
                continue

            if name_lc in IGNORED_GATES:
                new_circ.append(instr, qargs, cargs)
                continue

            if name_lc in UNSUPPORTED_GATES:
                for t in q_indices:
                    table.set_top(t)
                new_circ.append(instr, qargs, cargs)
                continue

            if name_lc == MEASURE_NAME:
                if q_indices:
                    table.set_top(q_indices[0])
                new_circ.append(instr, qargs, cargs)
                continue

            if name_lc == RESET_NAME:
                if q_indices:
                    table.reset_state(q_indices[0])
                new_circ.append(instr, qargs, cargs)
                continue

            if not qargs:
                new_circ.append(instr, qargs, cargs)
                continue

            min_contr = cls._minimize_controls(table, instr, qargs)
            if min_contr is None:
                continue

            instr_min_contr, qargs_min_contr = min_contr
            if cls._is_no_op(table, instr_min_contr, qargs_min_contr, max_amplitudes):
                continue

            cls._apply_gate(table, instr_min_contr, qargs_min_contr, max_amplitudes)
            new_circ.append(instr_min_contr, qargs_min_contr, cargs)

        return table, new_circ

    @classmethod
    def optimize(
        cls,
        circuit: QuantumCircuit,
        max_amplitudes: int | None = None,
    ) -> QuantumCircuit:
        """Perform base constant-propagation on circuit.
        """

        t, propagated_circ = cls._propagate(circuit, max_amplitudes)

        dag = cls.optimizing_with_layers_based_on_states(propagated_circ)




        return dag_to_circuit(dag)


    @staticmethod
    def _split_ctrl_target(instr: Instruction, qargs: Sequence[Qubit]) -> tuple[tuple[int, ...], tuple[int, ...]]:

        if isinstance(instr, ControlledGate):
            nctrl = instr.num_ctrl_qubits
        else:
            nctrl = 0

        ctrl = tuple(q._index for q in qargs[:nctrl])
        trgt = tuple(q._index for q in qargs[nctrl:])
        return ctrl, trgt


    @classmethod
    def _snapshot_qubit_states(cls, table: UnionTable) -> dict:
        """
        Copy QubitStateOrTop for each qubit from UnionTable.
        ret dict: [qubit_index:QubitStateOrTop].
        """

        snapshot = {}
        seen: dict = {}
        for i in range(table.size()):
            reg = table[i]
            if reg.is_top():
                snapshot[i] = QubitStateOrTop()
            else:
                qs = reg.get_qubit_state()
                key = id(qs)
                if key not in seen:
                    seen[key] = QubitStateOrTop(qs.clone())
                snapshot[i] = seen[key]
        return snapshot

    @staticmethod
    def _clone_qstate_or_top(entry: QubitStateOrTop) -> QubitStateOrTop:
        if entry.is_top():
            return QubitStateOrTop()
        return QubitStateOrTop(entry.get_qubit_state().clone())

    @staticmethod
    def qubit_state_equal_up_to_global_phase(qs1: QubitState, qs2: QubitState, eps: float = EPS) -> bool:
        if qs1.get_n_qubits() != qs2.get_n_qubits():
            return False

        n1 = sum(abs(v) ** 2 for v in qs1.state.values())
        n2 = sum(abs(v) ** 2 for v in qs2.state.values())
        if abs(n1 - n2) > eps:
            return False

        keys = set(qs1.state.keys()) | set(qs2.state.keys())

        phase = None
        for k in keys:
            a = qs1.state.get(k, 0j)
            b = qs2.state.get(k, 0j)

            a_zero = abs(a) <= eps
            b_zero = abs(b) <= eps

            if a_zero and b_zero:
                continue
            if a_zero != b_zero:
                return False

            cand = a / b
            if phase is None:
                phase = cand
            else:
                if abs(a - phase * b) > eps:
                    return False

        return True

    @classmethod
    def _states_equal_for_qubit(cls, snap_a: dict, snap_b: dict, qubit: int) -> bool:

        a = snap_a[qubit]
        b = snap_b[qubit]
        if a.is_top() or b.is_top():
            return False

        return cls.qubit_state_equal_up_to_global_phase(
            a.get_qubit_state(),
            b.get_qubit_state(),
        )

    @staticmethod
    def _build_layers_from_dag(dag) -> list:
        """
        ret list[list[DAGOpNode]] — list[DAGOpNode] = layer.
        """
        node_layer: dict[DAGOpNode, int] = {}  # DAGOpNode:layer_idx

        nodes = list(dag.topological_op_nodes())

        for node in nodes:
            max_pred_layer = -1
            for pred in dag.quantum_predecessors(node):
                if isinstance(pred, DAGOpNode):
                    pred_layer = node_layer.get(pred, -1)
                    if pred_layer > max_pred_layer:
                        max_pred_layer = pred_layer
            node_layer[node] = max_pred_layer + 1

        num_layers = (max(node_layer.values()) + 1) if node_layer else 0
        layers: list[list[DAGOpNode]] = [[] for _ in range(num_layers)]
        for node in nodes:
            layers[node_layer[node]].append(node)

        return layers


    @classmethod
    def optimizing_with_layers_based_on_states(cls, circuit: QuantumCircuit):
        dag = circuit_to_dag(circuit)
        num_qubits = dag.num_qubits()

        def build_states_after(current_layers: list[list[DAGOpNode]]) -> list[dict]:
            """Build states_after[k]: state after executing first k layers."""
            table = UnionTable(num_qubits)
            out: list[dict] = [cls._snapshot_qubit_states(table)]

            for layer in current_layers:
                for node in layer:
                    instr = node.op
                    qargs = node.qargs
                    name_lc = instr.name.lower()

                    if name_lc in IGNORED_GATES:
                        continue
                    if name_lc in UNSUPPORTED_GATES:
                        for q in qargs:
                            table.set_top(q._index)
                        continue
                    if name_lc == MEASURE_NAME:
                        for q in qargs:
                            table.set_top(q._index)
                        continue
                    if name_lc == RESET_NAME:
                        for q in qargs:
                            table.reset_state(q._index)
                        continue

                    if not qargs:
                        continue

                    cls._apply_gate(table, instr, qargs, cls.MAX_AMPLITUDES)
                    # separate() the oversized remainder can live in another group.
                    cls._check_amplitudes(table, cls.MAX_AMPLITUDES)
                out.append(cls._snapshot_qubit_states(table))
            return out

        def nodes_in_window(current_layers: list[list[DAGOpNode]], left: int, right: int) -> set[DAGOpNode]:
            nodes: set[DAGOpNode] = set()
            for layer_idx in range(left, right + 1):
                for node in current_layers[layer_idx]:
                    nodes.add(node)
            return nodes

        def window_removals(
            current_layers: list[list[DAGOpNode]],
            current_states_after: list[dict],
            left: int,
            right: int,
        ) -> set[DAGOpNode]:
            snap_start = current_states_after[left]
            snap_end = current_states_after[right + 1]
            stable_qubits = {
                q: cls._states_equal_for_qubit(snap_start, snap_end, q)
                for q in range(num_qubits)
            }

            candidate_nodes = nodes_in_window(current_layers, left, right)
            nodes_to_remove: set[DAGOpNode] = set()
            for node in candidate_nodes:
                touched_qubits = [q._index for q in node.qargs]
                if touched_qubits and all(stable_qubits[q] for q in touched_qubits):
                    nodes_to_remove.add(node)

            def entanglement_group(q: int) -> set[int]:
                """All qubits sharing qubitState at the
                window boundary. Change affects state, so a
                gate cannot be removed independently of gates on them"""
                group = {q}
                for snap in (snap_start, snap_end):
                    state = snap.get(q)
                    if state is None or state.is_top():
                        continue
                    for j in range(num_qubits):
                        other = snap.get(j)
                        if other is not None and not other.is_top() and other is state:
                            group.add(j)
                return group

            protected_qubits = set()
            for node in candidate_nodes:
                if len(node.qargs) > 1 and node not in nodes_to_remove:
                    for q in node.qargs:
                        protected_qubits |= entanglement_group(q._index)
            changed = True
            while changed:
                changed = False
                for node in list(nodes_to_remove):
                    touched = [q._index for q in node.qargs]
                    if any(q in protected_qubits for q in touched):
                        nodes_to_remove.remove(node)
                        for q in node.qargs:
                            protected_qubits |= entanglement_group(q._index)
                        changed = True


            return nodes_to_remove

        layers = cls._build_layers_from_dag(dag)

        made_progress = True
        while made_progress:
            made_progress = False
            states_after = build_states_after(layers)

            i = 1
            while i < len(layers):
                left = i - 1
                right = i
                nodes_to_remove = window_removals(layers, states_after, left, right)
                if not nodes_to_remove:
                    i += 1
                    continue

                focus_qubits: set[int] = set()
                for node in nodes_to_remove:
                        l = [q._index for q in node.qargs]
                        for x in l:
                            focus_qubits.add(x)

                if focus_qubits:
                    expanded = True
                    while expanded:
                        expanded = False

                        if left > 0 and right + 1 < len(layers):
                            if any(
                                cls._states_equal_for_qubit(states_after[left - 1], states_after[right + 2], q)
                                for q in focus_qubits
                            ):
                                right += 1
                                left -= 1
                                expanded = True

                        elif right + 1 < len(layers):
                            if any(
                                cls._states_equal_for_qubit(states_after[left], states_after[right + 2], q)
                                for q in focus_qubits
                            ):
                                right += 1
                                expanded = True

                        elif left > 0:
                            if any(
                                cls._states_equal_for_qubit(states_after[left - 1], states_after[right + 1], q)
                                for q in focus_qubits
                            ):
                                left -= 1
                                expanded = True


                    nodes_to_remove |= window_removals(layers, states_after, left, right)
                    if not nodes_to_remove:
                        i += 1
                        continue

                removed_any = False
                for node in nodes_to_remove:
                    try:
                        dag.remove_op_node(node)
                        removed_any = True
                        #print(f"Removing node {node}")
                    except Exception as e:
                        print(f"Couldn't delete the node {node}: {e}")

                if removed_any:
                    layers = cls._build_layers_from_dag(dag)
                    made_progress = True
                    break

                i += 1




        return dag

    @staticmethod
    def _control_polarity(instr: Instruction):
        """Required control value for a gate.

        Returns True  for standard controls (activate on all-ones, the default),
                False for open controls (activate on all-zeros, ctrl_state=0),
                None  for any mixed control state (not precisely supported).
        """
        if not isinstance(instr, ControlledGate):
            return True
        nctrl = instr.num_ctrl_qubits
        cs = instr.ctrl_state
        if cs == (1 << nctrl) - 1:
            return True
        if cs == 0:
            return False
        return None

    @classmethod
    def _minimize_controls(cls, table: UnionTable, instr: Instruction, qargs: Sequence[Qubit]):
        q_indices = [q._index for q in qargs]
        if isinstance(instr, ControlledGate):
            nctrl = instr.num_ctrl_qubits
            controls: List[int] = q_indices[:nctrl]
        else:
            controls = []

        polarity = cls._control_polarity(instr)
        if polarity is None:
            # Mixed control state: keep the gate as-is; its effect on the abstract
            # state is handled conservatively in _apply_gate (all qubits -> TOP).
            return (instr, qargs)

        activation, min_controls = table.minimize_controls(controls, control_value=polarity)
        if activation == ActivationState.NEVER:
            return None

        instr_eff, qargs_eff = cls._rebuild_instruction(instr, qargs, min_controls, polarity)
        return (instr_eff, qargs_eff)

    @classmethod
    def _apply_gate(cls, table: UnionTable, instr: Instruction, qargs: Sequence[Qubit], max_amplitudes: int) -> None:
        q_indices = [q._index for q in qargs]
        if isinstance(instr, ControlledGate):
            nctrl = instr.num_ctrl_qubits
            controls: List[int] = q_indices[:nctrl]
            targets: List[int] = q_indices[nctrl:]
        else:
            controls = []
            targets = q_indices

        polarity = cls._control_polarity(instr)
        if polarity is None:
            # Mixed control state: cannot be represented exactly -> lose info soundly.
            for t in q_indices:
                table.set_top(t)
            return

        if not targets:
            return

        if len(targets) == 1:
            cls._apply_single_qubit_gate(table, targets[0], controls, instr, polarity)
        elif len(targets) == 2:
            cls._apply_two_qubit_gate(table, targets[0], targets[1], controls, instr, polarity)
        elif len(targets) == 3:
            cls._apply_three_qubit_gate(table, targets[0], targets[1], targets[2], controls, instr, polarity)
        elif len(targets) == 4:
            cls._apply_four_qubit_gate(table, targets[0], targets[1], targets[2], targets[3], controls, instr, polarity)
        else:
            for t in q_indices:
                table.set_top(t)

        cls._check_amplitude(table, max_amplitudes, targets[0])

    @staticmethod
    def _apply_single_qubit_gate(table: UnionTable, target: int, controls: Sequence[int], instr: Instruction, control_value: bool = True) -> None:
        table.combine(target, list(controls))
        if table.is_top(target):
            return
        idx_t = table.index_in_state(target)
        idx_ctrl = table.index_in_state_list(list(controls))
        matrix = _single_qubit_matrix(instr)
        table[target].get_qubit_state().apply_gate(idx_t, matrix, idx_ctrl, control_value)

        table.separate(target)
        for c in controls:
            table.separate(c)

    @staticmethod
    def _apply_two_qubit_gate(table: UnionTable, t1: int, t2: int, controls: Sequence[int], instr: Instruction, control_value: bool = True) -> None:
        table.combine(t1, list(controls))
        table.combine(t1, t2)
        if table.is_top(t1):
            return
        idx1 = table.index_in_state(t1)
        idx2 = table.index_in_state(t2)
        idx_ctrl = table.index_in_state_list(list(controls))
        matrix = _two_qubit_matrix(instr)
        table[t1].get_qubit_state().apply_two_qubit_gate(idx1, idx2, matrix, idx_ctrl, control_value)

        table.separate(t1)
        table.separate(t2)
        for c in controls:
            table.separate(c)

    @staticmethod
    def _apply_three_qubit_gate(table: UnionTable, t1: int, t2: int, t3: int, controls: Sequence[int], instr: Instruction, control_value: bool = True) -> None:
        table.combine(t1, list(controls))
        table.combine(t1, t2)
        table.combine(t1, t3)
        if table.is_top(t1):
            return
        idx1 = table.index_in_state(t1)
        idx2 = table.index_in_state(t2)
        idx3 = table.index_in_state(t3)
        idx_ctrl = table.index_in_state_list(list(controls))
        matrix = _three_qubit_matrix(instr)
        table[t1].get_qubit_state().apply_three_qubit_gate(idx1, idx2, idx3, matrix, idx_ctrl, control_value)

        table.separate(t1)
        table.separate(t2)
        table.separate(t3)
        for c in controls:
            table.separate(c)

    @staticmethod
    def _apply_four_qubit_gate(table: UnionTable, t1: int, t2: int, t3: int, t4: int, controls: Sequence[int], instr: Instruction, control_value: bool = True) -> None:
        table.combine(t1, list(controls))
        table.combine(t1, t2)
        table.combine(t1, t3)
        table.combine(t1, t4)
        if table.is_top(t1):
            return
        idx1 = table.index_in_state(t1)
        idx2 = table.index_in_state(t2)
        idx3 = table.index_in_state(t3)
        idx4 = table.index_in_state(t4)
        idx_ctrl = table.index_in_state_list(list(controls))
        matrix = _four_qubit_matrix(instr)
        table[t1].get_qubit_state().apply_four_qubit_gate(idx1, idx2, idx3, idx4, matrix, idx_ctrl, control_value)

        table.separate(t1)
        table.separate(t2)
        table.separate(t3)
        table.separate(t4)
        for c in controls:
            table.separate(c)

    @staticmethod
    def _rebuild_instruction(instr: Instruction, qargs: List[Qubit], min_controls: List[int], control_value: bool = True) -> Tuple[Instruction, List[Qubit]]:
        """Return (instruction, qargs) with the pruned control set.

        `control_value` is the polarity of the kept controls: True for standard
        (all-ones) controls, False for all-zeros controls.
        """
        if not isinstance(instr, ControlledGate):
            return instr, qargs

        original_ctrls = instr.num_ctrl_qubits
        if len(min_controls) == original_ctrls:
            return instr, qargs

        base_gate: Gate = instr.base_gate
        new_ctrl_count = len(min_controls)
        if new_ctrl_count == 0:
            new_gate: Gate = base_gate
        else:
            new_ctrl_state = None if control_value else 0
            new_gate = base_gate.control(new_ctrl_count, ctrl_state=new_ctrl_state)

        ctrl_qubits = [q for q in qargs[:original_ctrls] if q._index in min_controls]
        target_qubits = qargs[original_ctrls:]
        new_qargs = ctrl_qubits + list(target_qubits)

        return new_gate, new_qargs
