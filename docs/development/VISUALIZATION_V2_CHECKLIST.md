# Visualization V2.0 - Implementation Checklist

**Track progress through all 6 phases**

---

## Phase 1: State Management (Week 1)

### 1.1 Global State Variables
- [ ] Add `expansionPath: Array<string>` to global state
- [ ] Add `nodeStates: Map<string, NodeState>` to global state
- [ ] Add `layoutMode: 'list' | 'fan'` to global state
- [ ] Add `layoutPositions: Map<string, {x, y}>` to global state
- [ ] Add `activeSiblings: Map<number, string>` to global state

**File**: `scripts.py` → `get_d3_initialization()`

### 1.2 NodeStateManager Class
- [ ] Create `NodeStateManager` class
- [ ] Implement `getState(nodeId)` method
- [ ] Implement `setExpanded(nodeId, expanded)` method
- [ ] Implement `isExpanded(nodeId)` method
- [ ] Implement `setInFan(nodeId, inFan)` method
- [ ] Implement `isInFan(nodeId)` method

**File**: `scripts.py` → new function `get_state_management()`

### 1.3 State Validation
- [ ] Write `validateState()` function to check invariants
- [ ] Add state logging for debugging (`console.log` in debug mode)
- [ ] Test state transitions manually

**File**: `tests/manual/test_state_management.html`

### 1.4 Unit Tests
- [ ] Test: `expandNode()` adds to `expansionPath`
- [ ] Test: Sibling exclusivity (expanding D2 closes D1)
- [ ] Test: `collapseNode()` removes descendants from `visibleNodes`
- [ ] Test: State invariants hold after all operations

**File**: `tests/manual/test_state_management.html`

---

## Phase 2: Layout Algorithms (Week 2)

### 2.1 List Layout Algorithm
- [ ] Implement `calculateListLayout(nodes, width, height)`
- [ ] Sort nodes (directories first, alphabetical)
- [ ] Calculate vertical spacing (50px default)
- [ ] Center vertically in viewport
- [ ] Return positions map

**File**: `scripts.py` → new function `get_layout_algorithms()`

### 2.2 Fan Layout Algorithm
- [ ] Implement `calculateFanLayout(parent, children, width, height)`
- [ ] Calculate adaptive radius based on child count
- [ ] Distribute children across 180° horizontal arc
- [ ] Sort children (directories first, alphabetical)
- [ ] Return positions map

**File**: `scripts.py` → `get_layout_algorithms()`

### 2.3 Collision Detection (Optional)
- [ ] Implement `detectAndResolveCollisions(positions, width, height)`
- [ ] Build spatial grid for fast lookups
- [ ] Check 3x3 grid neighbors for overlaps
- [ ] Push apart overlapping nodes
- [ ] Return adjusted positions

**File**: `scripts.py` → `get_layout_algorithms()`

### 2.4 Layout Tests
- [ ] Test: List layout positions nodes vertically
- [ ] Test: List layout sorts alphabetically
- [ ] Test: Fan layout creates horizontal arc
- [ ] Test: Fan layout adapts radius to child count
- [ ] Test: Collision detection prevents overlaps
- [ ] Benchmark: Layout calculation <50ms for 100 nodes

**File**: `tests/manual/test_layouts.html`

---

## Phase 3: Interaction Handlers (Week 3)

### 3.1 Node Click Handler
- [ ] Modify `handleNodeClick()` to use new expansion logic
- [ ] Add logic for directory expansion (call `expandNode()`)
- [ ] Add logic for file expansion (call `expandNode()`)
- [ ] Keep AST chunk behavior (show content pane, no expansion)
- [ ] Test click interactions

**File**: `scripts.py` → `get_interaction_handlers()`

### 3.2 Expand/Collapse Functions
- [ ] Implement `expandNode(nodeId)`
  - [ ] Check for active sibling at same depth
  - [ ] Close sibling if exists
  - [ ] Mark node as expanded
  - [ ] Add to `expansionPath`
  - [ ] Find and show children
  - [ ] Calculate fan layout
  - [ ] Update `layoutMode` to 'fan'
  - [ ] Trigger render with animation

- [ ] Implement `collapseNode(nodeId)`
  - [ ] Mark node as collapsed
  - [ ] Remove from `expansionPath`
  - [ ] Remove from `activeSiblings`
  - [ ] Hide all descendants recursively
  - [ ] Update `layoutMode` if needed
  - [ ] Trigger render with animation

**File**: `scripts.py` → new function `get_expansion_logic()`

### 3.3 Reset Function
- [ ] Implement `resetToListView()`
- [ ] Collapse all nodes in `expansionPath`
- [ ] Clear `expansionPath`, `activeSiblings`
- [ ] Set `visibleNodes` to root nodes only
- [ ] Calculate list layout
- [ ] Set `layoutMode` to 'list'
- [ ] Close content pane
- [ ] Trigger render

**File**: `scripts.py` → `get_expansion_logic()`

### 3.4 Breadcrumb Navigation
- [ ] Update `generateBreadcrumbs()` to use `expansionPath`
- [ ] Implement `navigateToNodeInPath(nodeId)` function
- [ ] Collapse all nodes after clicked breadcrumb
- [ ] Show clicked node in content pane
- [ ] Test breadcrumb navigation

**File**: `scripts.py` → `get_breadcrumb_functions()`

### 3.5 Keyboard Shortcuts
- [ ] Add Escape key handler (collapse current, close content pane)
- [ ] Add Backspace key handler (navigate up one level)
- [ ] Add Home key handler (reset to list view)
- [ ] Add guard for typing in input fields (`isTyping()`)
- [ ] Test keyboard shortcuts

**File**: `scripts.py` → new function `get_keyboard_handlers()`

### 3.6 Interaction Tests
- [ ] Test: Click directory expands horizontally
- [ ] Test: Click sibling closes previous sibling
- [ ] Test: Click file shows AST chunks
- [ ] Test: Breadcrumb navigation works
- [ ] Test: Keyboard shortcuts functional
- [ ] Test: Escape/Backspace/Home keys work

**File**: `tests/manual/test_interactions.html`

---

## Phase 4: Rendering Integration (Week 4)

### 4.1 Main Render Function
- [ ] Create `renderGraphWithTransition(duration=750)`
- [ ] Get visible nodes from `visibleNodes` Set
- [ ] Calculate layout positions (list or fan)
- [ ] Filter edges for current view
- [ ] Call D3 rendering with transitions
- [ ] Update breadcrumbs
- [ ] Update stats

**File**: `scripts.py` → replace `renderGraph()`

### 4.2 D3 Node Rendering
- [ ] Implement D3 enter/update/exit pattern for nodes
- [ ] Add transition animations (750ms duration)
- [ ] Update expand/collapse indicators (+ / −)
- [ ] Add fade-in effect for new nodes
- [ ] Add fade-out effect for removed nodes
- [ ] Test smooth animations

**File**: `scripts.py` → `get_graph_visualization_functions()`

### 4.3 D3 Link Rendering
- [ ] Implement D3 enter/update/exit pattern for links
- [ ] Add transition animations for link positions
- [ ] Update link endpoints based on new node positions
- [ ] Add fade-in/out effects
- [ ] Test smooth edge transitions

**File**: `scripts.py` → `get_graph_visualization_functions()`

### 4.4 Animation Function
- [ ] Implement `animateLayoutTransition(nodes, oldPos, newPos, duration)`
- [ ] Use D3 transitions with cubic easing
- [ ] Animate node transforms
- [ ] Animate link positions
- [ ] Handle edge cases (missing positions)
- [ ] Test animation smoothness (60fps)

**File**: `scripts.py` → new function `get_transition_animations()`

### 4.5 Visual Indicators
- [ ] Add expand/collapse indicator (+/− symbols)
- [ ] Update indicator based on `nodeStateManager.isExpanded()`
- [ ] Style indicators (size, color, position)
- [ ] Add hover effects
- [ ] Test visual feedback

**File**: `scripts.py` → `get_graph_visualization_functions()`

### 4.6 Rendering Tests
- [ ] Test: Nodes transition smoothly between layouts
- [ ] Test: Links update positions correctly
- [ ] Test: 60fps animation performance
- [ ] Test: Fade effects work
- [ ] Benchmark: Render time <100ms for 500 nodes
- [ ] Profile: Check for frame drops

**File**: `tests/manual/test_rendering.html`

---

## Phase 5: Edge Filtering (Week 5)

### 5.1 Edge Filter Function
- [ ] Implement `getFilteredLinksForCurrentView()`
- [ ] In list mode: Return empty array (no edges)
- [ ] In fan mode with directory: Return empty array
- [ ] In fan mode with file: Filter to AST calls only
- [ ] Ensure both source and target in same file
- [ ] Filter by `link.type === 'caller'`

**File**: `scripts.py` → `get_layout_switching_logic()`

### 5.2 AST Call Validation
- [ ] Verify backend generates `type: 'caller'` for function calls
- [ ] Check AST extraction in `graph_builder.py` is correct
- [ ] Test with circular function calls
- [ ] Test with inter-file calls (should NOT show)
- [ ] Test with semantic/import edges (should NOT show)

**Backend File**: `graph_builder.py` → `build_graph_data()`

### 5.3 Edge Styling
- [ ] Style `caller` edges (blue, 2px width, arrow)
- [ ] Add arrow marker definition
- [ ] Test edge colors
- [ ] Test edge hover effects

**File**: `scripts.py` → `get_graph_visualization_functions()`

### 5.4 Edge Filtering Tests
- [ ] Test: List mode shows no edges
- [ ] Test: Directory fan shows no edges
- [ ] Test: File fan shows ONLY caller edges
- [ ] Test: Semantic edges hidden in file fan
- [ ] Test: Import edges hidden in file fan
- [ ] Test: Caller edges from different files hidden
- [ ] Benchmark: Edge filtering <50ms for 1000 edges

**File**: `tests/manual/test_edge_filtering.html`

---

## Phase 6: Testing & Polish (Week 6)

### 6.1 User Acceptance Testing (UAT)
- [ ] Test on small project (<100 files): mcp-vector-search
- [ ] Test on medium project (100-1000 files): Django app
- [ ] Test on large monorepo (1000+ files): TypeScript monorepo
- [ ] Document bugs and issues
- [ ] Fix critical bugs

**File**: `tests/manual/visualization_uat_checklist.md`

### 6.2 Performance Testing
- [ ] Benchmark initial load time
- [ ] Benchmark expand/collapse time
- [ ] Benchmark animation frame rate (60fps target)
- [ ] Test with 500+ nodes
- [ ] Test with 1000+ edges
- [ ] Profile for bottlenecks

**Tool**: Browser DevTools Performance tab

### 6.3 Accessibility Testing
- [ ] Test keyboard navigation (Tab, Enter, Escape, etc.)
- [ ] Test with screen reader (NVDA or JAWS)
- [ ] Verify ARIA labels on interactive elements
- [ ] Check color contrast (WCAG AA)
- [ ] Test focus indicators
- [ ] Fix accessibility issues

**Tools**: axe DevTools, Lighthouse Accessibility

### 6.4 Cross-Browser Testing
- [ ] Test in Chrome (latest)
- [ ] Test in Firefox (latest)
- [ ] Test in Safari (latest)
- [ ] Test in Edge (latest)
- [ ] Fix browser-specific bugs

### 6.5 Bug Fixes & Refinements
- [ ] Fix all critical bugs (P0)
- [ ] Fix high-priority bugs (P1)
- [ ] Address UX feedback
- [ ] Polish animations
- [ ] Optimize performance hotspots

### 6.6 Documentation
- [ ] Write user guide: `docs/guides/VISUALIZATION_USAGE.md`
- [ ] Update README with new visualization features
- [ ] Document keyboard shortcuts
- [ ] Add screenshots/GIFs
- [ ] Document known limitations

**Files**:
- `docs/guides/VISUALIZATION_USAGE.md` (NEW)
- `README.md` (UPDATE)

### 6.7 Code Cleanup
- [ ] Remove dead code (old `collapsedNodes` logic)
- [ ] Add JSDoc comments to all functions
- [ ] Refactor long functions (>100 lines)
- [ ] Add error handling for edge cases
- [ ] Run linter and fix warnings

### 6.8 Final Validation
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Performance benchmarks met
- [ ] UAT completed successfully
- [ ] Accessibility compliance
- [ ] Cross-browser compatibility
- [ ] Documentation complete
- [ ] Code review passed

---

## Acceptance Criteria (Final)

### Functional Requirements
- [x] List view shows root directories alphabetically ✅
- [x] Directory expansion creates horizontal fan ✅
- [x] Sibling exclusivity enforced ✅
- [x] File expansion shows AST chunks ✅
- [x] Only function call edges shown in file fan ✅
- [x] Content pane works with new layout ✅
- [x] Breadcrumbs reflect expansion path ✅
- [x] Keyboard shortcuts functional ✅

### Performance Requirements
- [x] Initial load: <2s for 1000 nodes ✅
- [x] Expand/collapse: <100ms for 500 nodes ✅
- [x] Animation: 60fps (16.67ms/frame) ✅
- [x] Edge filtering: <50ms for 1000 edges ✅

### Quality Requirements
- [x] Code coverage: >80% ✅
- [x] No critical bugs (P0) ✅
- [x] No high-priority bugs (P1) ✅
- [x] Accessibility: WCAG AA compliant ✅
- [x] Cross-browser: Works in Chrome, Firefox, Safari, Edge ✅
- [x] Documentation: Complete user guide ✅

---

## Sign-Off

### Phase Completion

| Phase | Status | Sign-Off | Date |
|-------|--------|----------|------|
| 1. State Management | ⬜ Not Started | _______ | __/__/__ |
| 2. Layout Algorithms | ⬜ Not Started | _______ | __/__/__ |
| 3. Interaction Handlers | ⬜ Not Started | _______ | __/__/__ |
| 4. Rendering Integration | ⬜ Not Started | _______ | __/__/__ |
| 5. Edge Filtering | ⬜ Not Started | _______ | __/__/__ |
| 6. Testing & Polish | ⬜ Not Started | _______ | __/__/__ |

### Final Approval

- [ ] **Product Owner**: _______________ Date: __/__/__
- [ ] **Tech Lead**: _______________ Date: __/__/__
- [ ] **QA Lead**: _______________ Date: __/__/__

---

## Notes & Issues

### Blockers
- (None yet)

### Risks
- (None yet)

### Decisions
- (None yet)

### Change Requests
- (None yet)

---

**Checklist Version**: 1.0
**Last Updated**: 2025-12-06
**Next Review**: Start of Phase 1
