# Visualization Architecture V2.0 - Executive Summary

**Quick Reference Guide**

## What's Changing?

### Current State (Force-Directed Graph)
- All descendants shown when expanding node
- Force simulation positions nodes dynamically
- Multiple relationship types visible simultaneously
- Spatial confusion as nodes move

### New State (List-Based Hierarchical Navigation)
- **List View**: Vertical alphabetical list at root
- **Horizontal Fan**: Children spread horizontally when expanded
- **Sibling Exclusivity**: Only one sibling path visible at a time
- **AST-Only Edges**: Function calls shown, no implicit relationships

## Key Design Decisions

| Decision | Why | Trade-off |
|----------|-----|-----------|
| Vertical list for root | Alphabetical sorting is intuitive | More vertical space needed |
| Horizontal fan expansion | Natural "opening folder" metaphor | Animation complexity |
| Sibling exclusivity | Reduces clutter, maintains focus | Can't view multiple siblings simultaneously |
| AST calls only | Explicit dependencies are actionable | Miss implicit relationships |
| D3.js for layout | Already integrated, powerful | Learning curve |

## Architecture at a Glance

```
User Click → State Update → Layout Calculation → Edge Filtering → D3 Render
     ↓            ↓                ↓                    ↓              ↓
  Directory   expansion      calculateFan()      AST calls      Animate
   expand      Path[]          positions           only         750ms
```

## Implementation Phases (6 Weeks)

| Phase | Duration | Focus | Risk |
|-------|----------|-------|------|
| 1. State Management | 1 week | Add expansion path tracking | Low |
| 2. Layout Algorithms | 1 week | List + fan positioning | Medium |
| 3. Interaction Handlers | 1 week | Click + keyboard handlers | Low |
| 4. Rendering Integration | 1 week | D3 transitions + animations | **High** |
| 5. Edge Filtering | 1 week | AST-only relationships | Low |
| 6. Testing & Polish | 1 week | UAT + performance | Medium |

## Key Metrics

### Performance Targets
- Expand/collapse: <100ms
- Animation: 60fps (16.67ms/frame)
- Edge filtering: <50ms for 1000 edges

### Code Impact
- **+750 new lines** (algorithms, state management)
- **-150 removed lines** (old handlers)
- **Net: +600 lines**
- **1 primary file modified** (`scripts.py`)

## Critical Algorithms

### 1. List Layout (O(n))
```python
def calculateListLayout(nodes):
    sorted_nodes = sort_alphabetically(nodes)  # Dirs first, then files
    spacing = 50px
    for i, node in enumerate(sorted_nodes):
        position[node] = (x=100, y=start + i*spacing)
```

### 2. Fan Layout (O(n))
```python
def calculateFanLayout(parent, children):
    radius = adaptive_radius(len(children))  # 200-400px
    arc = 180°  # Horizontal (left to right)
    for i, child in enumerate(sorted(children)):
        angle = π - (i / (len-1)) * π  # Distribute across arc
        position[child] = parent + (radius * cos(angle), radius * sin(angle))
```

### 3. State Transitions
```
INITIAL (List)
    ↓ click dir
EXPANDED (Fan) - dir children shown
    ↓ click sibling
SWITCHED (Fan) - old children hidden, new shown
    ↓ click file
FILE_EXPANDED (Fan) - AST chunks shown with call edges
    ↓ click collapse
INITIAL (List)
```

## What Gets Preserved?

✅ **Keep existing features**:
- Content pane with code viewer
- Breadcrumb navigation
- Search functionality
- Force-directed layout (as alternate mode)
- Cytoscape layouts (Dagre, Circle)
- Edge type filters

❌ **Remove/Replace**:
- `collapsedNodes` Set (replaced by `nodeStates` Map)
- Force simulation for list/fan modes
- Semantic/import edges in fan view

## High-Risk Areas

### 🔴 **Risk 1: Animation Performance**
- **Problem**: 60fps with 500+ nodes challenging
- **Mitigation**: Use CSS transforms (GPU), batch DOM updates
- **Fallback**: Disable animations if >1000 nodes

### 🟡 **Risk 2: Complex State Management**
- **Problem**: Expansion path + sibling exclusivity may introduce bugs
- **Mitigation**: Comprehensive unit tests, state validation
- **Fallback**: "Reset View" button for recovery

### 🟡 **Risk 3: Edge Filtering**
- **Problem**: Correctly filtering AST calls within files
- **Mitigation**: Explicit `type: 'caller'` checks, file path matching
- **Fallback**: Edge type toggle for manual control

## Testing Strategy

### Unit Tests
- State transitions (expand, collapse, sibling switch)
- Layout algorithms (positions calculated correctly)
- Edge filtering (only AST calls shown)

### Integration Tests
- Directory navigation flow
- File exploration flow
- Breadcrumb navigation
- Keyboard shortcuts

### Performance Benchmarks
- Initial list render: <50ms (100 nodes)
- Expand directory: <100ms (50 children)
- Expand file (AST): <150ms (100 chunks)
- Animation: <16.67ms/frame (60fps)

### UAT Projects
1. **Small** (<100 files): mcp-vector-search
2. **Medium** (100-1000 files): Django project
3. **Large** (1000+ files): TypeScript monorepo

## Files to Modify

### Primary
- `/src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
  - Add: `get_state_management()`, `get_layout_algorithms()`, `get_expansion_logic()`
  - Modify: `get_d3_initialization()`, `get_interaction_handlers()`, `get_breadcrumb_functions()`

### Tests (New)
- `/tests/manual/test_state_management.html`
- `/tests/manual/test_layouts.html`
- `/tests/manual/visualization_uat_checklist.md`

### Docs (New)
- `/docs/guides/VISUALIZATION_USAGE.md`
- `/docs/development/LAYOUT_ALGORITHMS.md`

## Success Criteria

- [ ] List view shows root directories alphabetically
- [ ] Directory expansion creates horizontal fan
- [ ] Sibling exclusivity enforced (one path at a time)
- [ ] File expansion shows AST chunks with call edges only
- [ ] Smooth 60fps animations
- [ ] Performance: <100ms expand/collapse for 500 nodes
- [ ] Content pane works seamlessly with new layout
- [ ] Keyboard navigation functional (Escape, Backspace, Home)
- [ ] UAT passes on small, medium, and large projects

## Next Steps

1. **Review** this design with stakeholders
2. **Approve** proposed architecture
3. **Begin Phase 1**: State Management (1 week)
4. **Schedule** weekly progress reviews
5. **Prepare** test environments (3 UAT projects)

## Questions?

See full design document: `VISUALIZATION_ARCHITECTURE_V2.md`

- **Detailed algorithms**: Section 5 (Layout Engine)
- **State transitions**: Section 6 (State Management)
- **Risk mitigation**: Section 10 (Risk Analysis)
- **Test cases**: Section 11 (Testing Strategy)
- **ASCII diagrams**: Appendix A

---

**Document Version**: 1.0
**Last Updated**: 2025-12-06
**Status**: Design Review
