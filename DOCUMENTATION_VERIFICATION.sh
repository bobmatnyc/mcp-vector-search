#!/bin/bash

echo "================================================"
echo "AI Code Review Documentation - Verification"
echo "================================================"
echo ""

echo "📁 Files Created/Updated:"
echo ""

echo "1. Main Feature Documentation:"
if [ -f "docs/features/code-review.md" ]; then
    size=$(wc -c < docs/features/code-review.md)
    lines=$(wc -l < docs/features/code-review.md)
    echo "   ✅ docs/features/code-review.md (${size} bytes, ${lines} lines)"
else
    echo "   ❌ docs/features/code-review.md NOT FOUND"
fi
echo ""

echo "2. README.md Update:"
if grep -q "## 🔍 AI Code Review" README.md; then
    echo "   ✅ README.md contains AI Code Review section"
    section_lines=$(sed -n '/## 🔍 AI Code Review/,/## 📖 Documentation/p' README.md | wc -l)
    echo "   📊 Section size: ${section_lines} lines"
else
    echo "   ❌ README.md missing AI Code Review section"
fi
echo ""

echo "3. HyperDev Article:"
if [ -f ~/Duetto/repos/duetto-code-intelligence/docs/hyperdev-article-draft.md ]; then
    size=$(wc -c < ~/Duetto/repos/duetto-code-intelligence/docs/hyperdev-article-draft.md)
    lines=$(wc -l < ~/Duetto/repos/duetto-code-intelligence/docs/hyperdev-article-draft.md)
    echo "   ✅ hyperdev-article-draft.md (${size} bytes, ${lines} lines)"
else
    echo "   ❌ hyperdev-article-draft.md NOT FOUND"
fi
echo ""

echo "4. CHANGELOG.md Update:"
if grep -q "\[2.10.0\]" CHANGELOG.md; then
    echo "   ✅ CHANGELOG.md contains v2.10.0 section"
    section_lines=$(sed -n '/## \[2.10.0\]/,/## \[2.8.0\]/p' CHANGELOG.md | wc -l)
    echo "   📊 Section size: ${section_lines} lines"
else
    echo "   ❌ CHANGELOG.md missing v2.10.0 section"
fi
echo ""

echo "5. Documentation Summary:"
if [ -f "docs/DOCUMENTATION_SUMMARY.md" ]; then
    size=$(wc -c < docs/DOCUMENTATION_SUMMARY.md)
    lines=$(wc -l < docs/DOCUMENTATION_SUMMARY.md)
    echo "   ✅ docs/DOCUMENTATION_SUMMARY.md (${size} bytes, ${lines} lines)"
else
    echo "   ❌ docs/DOCUMENTATION_SUMMARY.md NOT FOUND"
fi
echo ""

echo "================================================"
echo "📊 Content Verification:"
echo "================================================"
echo ""

echo "Checking key sections in code-review.md:"
if [ -f "docs/features/code-review.md" ]; then
    grep -c "^## " docs/features/code-review.md | xargs -I {} echo "   • Top-level sections: {}"
    grep -c "^### " docs/features/code-review.md | xargs -I {} echo "   • Sub-sections: {}"
    grep -c '```' docs/features/code-review.md | awk '{print "   • Code blocks: " $1/2}' 
    grep -c '^|' docs/features/code-review.md | xargs -I {} echo "   • Table rows: {}"
fi
echo ""

echo "================================================"
echo "🔗 Cross-References:"
echo "================================================"
echo ""

echo "Links in README.md to code-review.md:"
grep -c "docs/features/code-review.md" README.md | xargs -I {} echo "   • References: {}"
echo ""

echo "Links in code-review.md to other docs:"
grep -c "\.\./.*\.md" docs/features/code-review.md | xargs -I {} echo "   • Internal links: {}"
echo ""

echo "================================================"
echo "✅ Verification Complete!"
echo "================================================"
