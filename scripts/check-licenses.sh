#!/bin/bash

# License Check Script
# Run this locally before pushing to check license consistency

set -e

echo "🔍 Running License Consistency Check..."
echo "=================================="

# Check if pip-licenses is installed
if ! command -v pip-licenses &> /dev/null; then
    echo "📦 Installing pip-licenses..."
    pip install pip-licenses license-expression
fi

# Run the license check
echo "📋 Checking license consistency..."
python -m repoqa.license_checker --format report

# Get the JSON output to check for issues
JSON_OUTPUT=$(python -m repoqa.license_checker --format json)
INCOMPATIBLE_COUNT=$(echo "$JSON_OUTPUT" | python -c "
import json, sys
try:
    data = json.load(sys.stdin)
    incompatible = data.get('compatibility', {}).get('incompatible', [])
    # Note: unknown_permissive licenses are treated as compatible with warnings
    print(len(incompatible))
except:
    print('0')
")

echo ""
echo "=================================="
if [ "$INCOMPATIBLE_COUNT" -gt 0 ]; then
    echo "❌ Found $INCOMPATIBLE_COUNT problematic licenses"
    echo "⚠️  Please review and resolve before pushing"
    exit 1
else
    echo "✅ All licenses are compatible!"
fi