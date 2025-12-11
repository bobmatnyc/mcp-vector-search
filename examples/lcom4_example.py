#!/usr/bin/env python3
"""Example demonstrating LCOM4 cohesion metric calculation.

This example shows how to use the LCOM4Calculator to measure class cohesion
by analyzing method-attribute relationships.
"""

from pathlib import Path

from mcp_vector_search.analysis.collectors.cohesion import LCOM4Calculator


def main():
    """Demonstrate LCOM4 cohesion analysis."""
    # Example 1: Cohesive class (LCOM4 = 1)
    cohesive_code = """
class CohesiveClass:
    def __init__(self):
        self.x = 0
        self.y = 0

    def method_a(self):
        return self.x + self.y

    def method_b(self):
        return self.x * self.y

    def method_c(self):
        self.x += 1
        self.y += 1
"""

    # Example 2: Incohesive class (LCOM4 = 2)
    incohesive_code = """
class IncohesiveClass:
    def method_a(self):
        return self.x + self.y

    def method_b(self):
        return self.x * self.y

    def method_c(self):
        return self.z + self.w

    def method_d(self):
        return self.z - self.w
"""

    # Example 3: Highly incohesive class (LCOM4 = 3)
    highly_incohesive_code = """
class HighlyIncohesiveClass:
    def group1_method_a(self):
        return self.x

    def group1_method_b(self):
        return self.x + 1

    def group2_method_a(self):
        return self.y

    def group2_method_b(self):
        return self.y * 2

    def group3_method(self):
        return self.z
"""

    calculator = LCOM4Calculator()

    # Analyze cohesive class
    print("=" * 60)
    print("Example 1: Cohesive Class")
    print("=" * 60)
    result1 = calculator.calculate_file_cohesion(Path("cohesive.py"), cohesive_code)
    if result1.classes:
        cohesion = result1.classes[0]
        print(f"Class: {cohesion.class_name}")
        print(f"LCOM4: {cohesion.lcom4}")
        print(f"Methods: {cohesion.method_count}")
        print(f"Attributes: {cohesion.attribute_count}")
        print(
            f"Interpretation: {'Perfect cohesion!' if cohesion.lcom4 == 1 else 'Poor cohesion'}"
        )
        print("\nMethod-Attribute Map:")
        for method, attrs in cohesion.method_attributes.items():
            print(f"  {method}: {sorted(attrs)}")
    print()

    # Analyze incohesive class
    print("=" * 60)
    print("Example 2: Incohesive Class (Two Groups)")
    print("=" * 60)
    result2 = calculator.calculate_file_cohesion(Path("incohesive.py"), incohesive_code)
    if result2.classes:
        cohesion = result2.classes[0]
        print(f"Class: {cohesion.class_name}")
        print(f"LCOM4: {cohesion.lcom4}")
        print(f"Methods: {cohesion.method_count}")
        print(f"Attributes: {cohesion.attribute_count}")
        print(
            f"Interpretation: Class should be split into {cohesion.lcom4} separate classes"
        )
        print("\nMethod-Attribute Map:")
        for method, attrs in cohesion.method_attributes.items():
            print(f"  {method}: {sorted(attrs)}")
        print("\nSuggested split:")
        print("  Class 1: method_a, method_b (working with x, y)")
        print("  Class 2: method_c, method_d (working with z, w)")
    print()

    # Analyze highly incohesive class
    print("=" * 60)
    print("Example 3: Highly Incohesive Class (Three Groups)")
    print("=" * 60)
    result3 = calculator.calculate_file_cohesion(
        Path("highly_incohesive.py"), highly_incohesive_code
    )
    if result3.classes:
        cohesion = result3.classes[0]
        print(f"Class: {cohesion.class_name}")
        print(f"LCOM4: {cohesion.lcom4}")
        print(f"Methods: {cohesion.method_count}")
        print(f"Attributes: {cohesion.attribute_count}")
        print(
            f"Interpretation: Class has {cohesion.lcom4} disconnected groups - needs refactoring!"
        )
        print("\nMethod-Attribute Map:")
        for method, attrs in cohesion.method_attributes.items():
            print(f"  {method}: {sorted(attrs)}")
    print()

    # File with multiple classes
    multiple_classes_code = """
class GoodClass:
    def foo(self):
        return self.x

    def bar(self):
        return self.x + 1

class BadClass:
    def method_a(self):
        return self.a

    def method_b(self):
        return self.b

    def method_c(self):
        return self.c
"""

    print("=" * 60)
    print("Example 4: File with Multiple Classes")
    print("=" * 60)
    result4 = calculator.calculate_file_cohesion(
        Path("multiple.py"), multiple_classes_code
    )
    print(f"Total classes: {len(result4.classes)}")
    print(f"Average LCOM4: {result4.avg_lcom4:.2f}")
    print(f"Max LCOM4: {result4.max_lcom4}")
    print()
    for cohesion in result4.classes:
        print(f"  {cohesion.class_name}: LCOM4={cohesion.lcom4}")
    print()

    print("=" * 60)
    print("LCOM4 Interpretation Guide")
    print("=" * 60)
    print("LCOM4 = 1: Perfect cohesion - all methods work together")
    print("LCOM4 = 2-3: Moderate cohesion issues - consider refactoring")
    print("LCOM4 > 3: Poor cohesion - class should be split")
    print()
    print("Best Practice: Aim for LCOM4 = 1 by ensuring methods share attributes")


if __name__ == "__main__":
    main()
