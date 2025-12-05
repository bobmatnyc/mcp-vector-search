# Visualization Enhancements - User Guide

**Version**: v0.14.8+
**Date**: December 4, 2025

---

## What's New? 🎉

Three powerful enhancements to make code visualization more accurate and interactive:

1. **Fixed Cycle Detection** - No more false warnings
2. **Code Linking** - Click to jump to definitions
3. **Python Syntax Highlighting** - Keywords stand out

---

## Enhancement 1: Accurate Circular Dependency Detection

### What Changed?

**Before**: You might see warnings like:
```
⚠ Found 15 circular dependencies
```
...but most were false positives (not real cycles).

**After**: Only TRUE circular dependencies are reported:
```
⚠ Found 3 circular dependencies
```

### What's a TRUE Circular Dependency?

A → B → C → A (functions calling each other in a loop)

### What's NOT a Circular Dependency?

```
    A
   / \
  B   C
   \ /
    D
```
This is just multiple paths to D, **not a cycle**.

### How to Use

1. Run visualization:
   ```bash
   mcp-vector-search visualize
   ```

2. If you see a circular dependency warning:
   ```
   ⚠ Found 2 circular dependencies
   ```

3. Look for red pulsing links in the graph - these are the cycles

4. **Red links** = Circular dependencies you should investigate

### Why It Matters

Circular dependencies can cause:
- Infinite recursion
- Tight coupling (hard to refactor)
- Module loading issues

Now you can **trust the warnings** - they're accurate!

---

## Enhancement 2: Click-to-Navigate Code Links

### What's This?

When viewing code, **function and class names are now clickable**. Click them to instantly jump to their definitions!

### How to Use

#### Step 1: Open Code Viewer
Click any function/class node in the graph to see its code in the right panel.

#### Step 2: Spot the Links
Function and class names appear with a **blue dotted underline**:

```python
def process_data(items):
    result = calculate_total(items)  # ← "calculate_total" is clickable
    return validate_result(result)   # ← "validate_result" is clickable
```

#### Step 3: Click to Navigate
Hover over a link → it turns **solid blue** → click it!

**What happens?**
1. The graph zooms to the target function
2. The target function's code appears in the viewer
3. The node is highlighted in the graph

### Visual Guide

**Link Appearance**:
- **Normal**: Blue text, dotted underline
- **Hover**: Brighter blue, solid underline
- **Tooltip**: "Jump to function_name"

### What's Linkable?

✅ **User-defined functions**: `def my_function()`
✅ **User-defined classes**: `class MyClass:`
✅ **Methods**: Inside classes

❌ **Python keywords**: `def`, `if`, `return` (bolded instead)
❌ **Built-in functions**: `print()`, `len()`, `str()` (bolded instead)
❌ **Self-references**: A function won't link to itself

### Example Workflow

```python
# You're viewing: main.py::run()
def run():
    config = load_config()  # ← Click here
    # ... jumps to config.py::load_config()

# Now viewing: config.py::load_config()
def load_config():
    data = parse_yaml(file)  # ← Click here
    # ... jumps to parser.py::parse_yaml()
```

**Pro Tip**: This is perfect for exploring unfamiliar codebases!

---

## Enhancement 3: Bold Python Keywords

### What's This?

Python keywords and built-in types are now **bold red** for instant recognition.

### What's Bolded?

#### Keywords (31 total)
```python
def, class, if, else, elif, for, while, return
import, from, try, except, finally, with, as
async, await, yield, lambda, pass, break, continue
raise, assert, del, global, nonlocal, is, in
and, or, not
```

#### Built-in Types & Functions (26 total)
```python
str, int, float, bool, list, dict, set, tuple
None, True, False
len, range, enumerate, zip, map, filter
sorted, reversed, any, all, isinstance
hasattr, getattr, setattr, print, type
```

### Example

**Before** (all same color):
```python
def calculate(x):
    if isinstance(x, int):
        return x * 2
    else:
        return None
```

**After** (keywords in bold red):
```python
**def** calculate(x):
    **if** **isinstance**(x, **int**):
        **return** x * 2
    **else**:
        **return** **None**
```

### Why It's Useful

- **Faster scanning**: Keywords jump out at you
- **Better readability**: Distinguish language features from your code
- **Consistent style**: Matches modern code editors

---

## Quick Tips

### Tip 1: Explore Code Relationships
1. Click a function in the graph
2. See its code in the right panel
3. Click linked functions to explore dependencies
4. Use "Reset View" button to go back to overview

### Tip 2: Track Down Circular Dependencies
1. Look for red pulsing links
2. Follow the cycle: A → B → C → A
3. Decide how to break the cycle (refactor, extract common code, etc.)

### Tip 3: Navigate Large Codebases
1. Start at a high-level function (like `main()`)
2. Click through linked functions to understand flow
3. Use the graph to see the big picture
4. Use the code viewer for details

---

## Keyboard Shortcuts (Unchanged)

| Action | Shortcut |
|--------|----------|
| Zoom in | Mouse wheel up |
| Zoom out | Mouse wheel down |
| Pan | Click + drag background |
| Reset view | Click "Reset View" button |
| Close code pane | Click × in top-right |

---

## Troubleshooting

### "Code links aren't working"
**Fix**: Make sure the target function is loaded in the graph. Links only work for nodes currently in the visualization.

### "I don't see bold keywords"
**Fix**: This only applies to Python code. JavaScript/TypeScript will be added in a future update.

### "Browser shows no links"
**Fix**: You need a modern browser (Chrome 62+, Firefox 78+, Safari 16.4+). Older browsers don't support the regex patterns used.

### "Circular dependency warnings seem wrong"
**Fix**: They shouldn't be! If you see a warning, there's a real cycle. Trace the red links to find it. If you think it's incorrect, please file a bug report.

---

## What's Next?

Future enhancements we're considering:
- 🎨 Multi-language support (JavaScript, TypeScript, Go, etc.)
- 📦 Import-to-module linking
- 🎬 Animated cycle visualization
- 🔍 Search term highlighting in code viewer
- 📋 Copy code with links preserved

---

## Feedback

Have suggestions or found a bug?
- **GitHub Issues**: Report bugs or request features
- **Discussions**: Share how you're using these features

---

## Summary

Three simple but powerful improvements:

1. **Trust the cycle warnings** - they're accurate now
2. **Click function names** - navigate instantly
3. **Keywords pop out** - read code faster

**Try it now**:
```bash
mcp-vector-search visualize
```

Enjoy exploring your codebase! 🚀

---

**Last Updated**: December 4, 2025
**Version**: v0.14.8+
