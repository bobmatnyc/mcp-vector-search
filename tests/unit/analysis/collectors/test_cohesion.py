"""Unit tests for LCOM4 cohesion metric collector."""

from pathlib import Path

from mcp_vector_search.analysis.collectors.cohesion import (
    ClassCohesion,
    FileCohesion,
    LCOM4Calculator,
    MethodAttributeAccess,
    UnionFind,
)


class TestUnionFind:
    """Test UnionFind data structure."""

    def test_initialization(self):
        """Test union-find initializes with independent items."""
        uf = UnionFind(["a", "b", "c"])
        assert uf.count_components() == 3

    def test_union_connects_items(self):
        """Test union operation connects two items."""
        uf = UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        assert uf.count_components() == 2

    def test_transitive_connection(self):
        """Test transitive connections: a-b-c all connected."""
        uf = UnionFind(["a", "b", "c"])
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.count_components() == 1
        assert uf.find("a") == uf.find("c")

    def test_multiple_components(self):
        """Test multiple disjoint components."""
        uf = UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")
        uf.union("c", "d")
        assert uf.count_components() == 2

    def test_empty_union_find(self):
        """Test empty union-find."""
        uf = UnionFind([])
        assert uf.count_components() == 0


class TestMethodAttributeAccess:
    """Test MethodAttributeAccess data class."""

    def test_initialization(self):
        """Test method attribute access initialization."""
        method = MethodAttributeAccess(method_name="foo", attributes={"x", "y"})
        assert method.method_name == "foo"
        assert method.attributes == {"x", "y"}

    def test_default_attributes(self):
        """Test default empty attribute set."""
        method = MethodAttributeAccess(method_name="bar")
        assert method.attributes == set()


class TestClassCohesion:
    """Test ClassCohesion data class."""

    def test_initialization(self):
        """Test class cohesion initialization."""
        cohesion = ClassCohesion(
            class_name="MyClass",
            lcom4=2,
            method_count=4,
            attribute_count=3,
            method_attributes={"foo": {"x"}, "bar": {"y"}},
        )
        assert cohesion.class_name == "MyClass"
        assert cohesion.lcom4 == 2
        assert cohesion.method_count == 4
        assert cohesion.attribute_count == 3


class TestFileCohesion:
    """Test FileCohesion data class."""

    def test_initialization(self):
        """Test file cohesion initialization."""
        cohesion = FileCohesion(
            file_path=Path("test.py"),
            classes=[],
            avg_lcom4=1.5,
            max_lcom4=2,
        )
        assert cohesion.file_path == Path("test.py")
        assert cohesion.avg_lcom4 == 1.5
        assert cohesion.max_lcom4 == 2


class TestLCOM4Calculator:
    """Test LCOM4Calculator."""

    def test_single_method_class_lcom4_is_one(self):
        """Class with one method should have LCOM4=1."""
        code = """
class SingleMethod:
    def method_a(self):
        return self.x + self.y
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        assert result.classes[0].lcom4 == 1
        assert result.classes[0].method_count == 1

    def test_cohesive_class_lcom4_is_one(self):
        """Methods sharing attributes should have LCOM4=1."""
        code = """
class CohesiveClass:
    def method_a(self):
        return self.x + self.y

    def method_b(self):
        return self.x * self.y

    def method_c(self):
        return self.x - self.y
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.lcom4 == 1, "All methods share x and y"
        assert cohesion.method_count == 3
        assert cohesion.attribute_count == 2

    def test_incohesive_class_lcom4_greater_than_one(self):
        """Disjoint method groups should have LCOM4>1."""
        code = """
class DisjointMethods:
    def method_a(self):
        return self.x + self.y

    def method_b(self):
        return self.x * self.y

    def method_c(self):
        return self.z + self.w

    def method_d(self):
        return self.z - self.w
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.lcom4 == 2, "Two groups: (a,b) and (c,d)"
        assert cohesion.method_count == 4
        assert cohesion.attribute_count == 4

    def test_class_with_no_attributes_lcom4_equals_method_count(self):
        """Each method with no shared attributes is its own component."""
        code = """
class NoSharedAttributes:
    def method_a(self):
        return self.a

    def method_b(self):
        return self.b

    def method_c(self):
        return self.c
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.lcom4 == 3, "Each method is independent"
        assert cohesion.method_count == 3

    def test_transitive_cohesion(self):
        """Transitive connections: A-B share x, B-C share y -> all connected."""
        code = """
class TransitiveConnection:
    def method_a(self):
        return self.x

    def method_b(self):
        return self.x + self.y

    def method_c(self):
        return self.y
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.lcom4 == 1, "All methods transitively connected"
        assert cohesion.method_count == 3

    def test_empty_class_lcom4_is_zero(self):
        """Class with no methods should have LCOM4=0."""
        code = """
class EmptyClass:
    pass
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        # Empty class is skipped
        assert len(result.classes) == 0

    def test_multiple_classes_in_file(self):
        """Multiple classes each have their own LCOM4."""
        code = """
class ClassA:
    def foo(self):
        return self.x

    def bar(self):
        return self.x


class ClassB:
    def baz(self):
        return self.a

    def qux(self):
        return self.b
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 2
        assert result.classes[0].class_name == "ClassA"
        assert result.classes[0].lcom4 == 1
        assert result.classes[1].class_name == "ClassB"
        assert result.classes[1].lcom4 == 2

    def test_static_methods_excluded(self):
        """@staticmethod should not access self, so excluded."""
        code = """
class WithStatic:
    def instance_method(self):
        return self.x

    @staticmethod
    def static_method():
        return 42
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        # Static method has no self access, so only instance_method counted
        assert cohesion.lcom4 == 1
        assert cohesion.method_count == 2  # Both methods exist
        assert len(cohesion.method_attributes) == 1  # Only instance_method tracked

    def test_class_methods_excluded(self):
        """@classmethod uses cls, not self, so excluded."""
        code = """
class WithClassMethod:
    def instance_method(self):
        return self.x

    @classmethod
    def class_method(cls):
        return cls.__name__
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.method_count == 2
        assert len(cohesion.method_attributes) == 1

    def test_property_methods_included(self):
        """@property methods should be included if they access self."""
        code = """
class WithProperty:
    @property
    def prop_a(self):
        return self.x

    def method_b(self):
        return self.x + self.y
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.lcom4 == 1, "Property and method share x"
        assert cohesion.method_count == 2

    def test_method_with_no_self_accesses(self):
        """Methods that don't access self.* are ignored in graph."""
        code = """
class NoSelfAccess:
    def method_a(self):
        return self.x

    def method_b(self):
        # No self.* access
        return 42

    def method_c(self):
        return self.x
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        # method_b not in graph, a and c connected via x
        assert cohesion.lcom4 == 1

    def test_aggregate_metrics(self):
        """Test avg_lcom4 and max_lcom4 calculation."""
        code = """
class ClassA:
    def foo(self):
        return self.x

    def bar(self):
        return self.x


class ClassB:
    def baz(self):
        return self.a

    def qux(self):
        return self.b

    def xyz(self):
        return self.c
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 2
        assert result.avg_lcom4 == (1 + 3) / 2  # (ClassA=1, ClassB=3) / 2
        assert result.max_lcom4 == 3

    def test_complex_attribute_expressions(self):
        """Test self.attr in complex expressions."""
        code = """
class ComplexExpressions:
    def method_a(self):
        result = self.x + 10
        if self.y > 0:
            return result * self.y
        return result

    def method_b(self):
        return [self.x, self.y, self.z]
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.lcom4 == 1, "Both methods share x and y"
        assert cohesion.attribute_count == 3  # x, y, z

    def test_nested_classes_treated_separately(self):
        """Nested classes should be treated as separate entities."""
        code = """
class Outer:
    def outer_method(self):
        return self.x

    class Inner:
        def inner_method(self):
            return self.y
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        # Both outer and inner classes should be found
        assert len(result.classes) == 2
        class_names = {c.class_name for c in result.classes}
        assert "Outer" in class_names
        assert "Inner" in class_names

    def test_method_names_extracted_correctly(self):
        """Test method names are correctly extracted."""
        code = """
class TestNames:
    def method_one(self):
        return self.x

    def method_two(self):
        return self.x
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert "method_one" in cohesion.method_attributes
        assert "method_two" in cohesion.method_attributes

    def test_only_instance_methods_counted(self):
        """Only methods accessing instance attributes counted in LCOM4 graph."""
        code = """
class MixedMethods:
    def with_self(self):
        return self.x

    def without_self(self):
        local_var = 10
        return local_var

    @staticmethod
    def static():
        return 5

    @classmethod
    def clsmethod(cls):
        return cls.__name__
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        # Only with_self in attribute graph
        assert len(cohesion.method_attributes) == 1
        assert "with_self" in cohesion.method_attributes

    def test_init_method_included(self):
        """__init__ should be included if it accesses self."""
        code = """
class WithInit:
    def __init__(self):
        self.x = 10
        self.y = 20

    def use_x(self):
        return self.x
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.lcom4 == 1, "__init__ and use_x connected via x"
        assert cohesion.method_count == 2

    def test_dunder_methods_included(self):
        """Dunder methods like __str__ should be included."""
        code = """
class WithDunder:
    def __str__(self):
        return str(self.x)

    def get_x(self):
        return self.x
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.lcom4 == 1, "__str__ and get_x connected via x"

    def test_file_with_no_classes(self):
        """File with no classes should return empty result."""
        code = """
def standalone_function():
    return 42

x = 10
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 0
        assert result.avg_lcom4 == 0.0
        assert result.max_lcom4 == 0

    def test_three_component_class(self):
        """Class with three disconnected groups."""
        code = """
class ThreeGroups:
    def group1_a(self):
        return self.x

    def group1_b(self):
        return self.x + 1

    def group2_a(self):
        return self.y

    def group2_b(self):
        return self.y * 2

    def group3_a(self):
        return self.z
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.lcom4 == 3, "Three disconnected groups"
        assert cohesion.method_count == 5

    def test_attribute_assignment_counted(self):
        """Attribute assignment (self.x = ...) should be counted."""
        code = """
class WithAssignment:
    def setter_a(self):
        self.x = 10

    def getter_a(self):
        return self.x
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.lcom4 == 1, "Both methods access x"

    def test_method_chaining_self_attributes(self):
        """Test self.attr.method() patterns."""
        code = """
class MethodChaining:
    def method_a(self):
        return self.x.upper()

    def method_b(self):
        return self.x.lower()
"""
        calculator = LCOM4Calculator()
        result = calculator.calculate_file_cohesion(Path("test.py"), code)

        assert len(result.classes) == 1
        cohesion = result.classes[0]
        assert cohesion.lcom4 == 1, "Both methods access self.x"
