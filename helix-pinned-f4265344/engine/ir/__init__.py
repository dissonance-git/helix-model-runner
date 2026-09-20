"""Provider-neutral intermediate representation, transformations, and lowering."""

from .lowering import LoweringRegistry, TargetAdapter, lower_operation, raise_result
from .model import IR_VERSION, build_module, build_operation, topological_order, validate_module
from .passes import (
    DEFECT_VERSION,
    TRANSFORMATION_VERSION,
    TRANSFORM_OPERATOR_BASIS,
    TRANSFORM_OPERATOR_CONTRACTS,
    TRANSFORM_STATES,
    OperatorSpec,
    TransformationRegistry,
    commutation_comparison,
    composition,
    execute_transform_program,
    execute_transformation,
    require_guarded_exact_result_verification,
    transform_operator_contract,
    verify_operator_preconditions,
)

__all__ = [
    "DEFECT_VERSION",
    "IR_VERSION",
    "LoweringRegistry",
    "OperatorSpec",
    "TRANSFORMATION_VERSION",
    "TRANSFORM_OPERATOR_BASIS",
    "TRANSFORM_OPERATOR_CONTRACTS",
    "TRANSFORM_STATES",
    "TargetAdapter",
    "TransformationRegistry",
    "build_module",
    "build_operation",
    "commutation_comparison",
    "composition",
    "execute_transform_program",
    "execute_transformation",
    "lower_operation",
    "raise_result",
    "require_guarded_exact_result_verification",
    "topological_order",
    "transform_operator_contract",
    "validate_module",
    "verify_operator_preconditions",
]
