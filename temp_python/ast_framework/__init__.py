from temp_python.ast_framework.ast_node_type import ASTNodeType  # noqa: F401
from temp_python.ast_framework.ast_node import ASTNode  # noqa: F401
from temp_python.ast_framework.ast import AST  # noqa: F401

# register all standard computed fields from 'computed_fields_catalog'
from temp_python.ast_framework.computed_fields_catalog.standard_fields import (
    register_standard_computed_properties,
)

register_standard_computed_properties()
