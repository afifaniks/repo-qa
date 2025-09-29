# License Policy Quick Reference

This document provides a quick reference for our OSS Component Usage Policy. For the complete policy, see [CONTRIBUTING.md](../CONTRIBUTING.md#oss-component-usage-policy).

## Allowed Licenses

| License | Type | Notes |
|---------|------|-------|
| MIT | Permissive | Preferred - same as our project |
| Apache-2.0 | Permissive | Includes patent grants |
| BSD-2-Clause | Permissive | Simple, compatible |
| BSD-3-Clause | Permissive | Includes non-endorsement clause |
| ISC | Permissive | Simplified BSD |

## Prohibited Licenses

| License | Type | Reason |
|---------|------|--------|
| GPL-2.0, GPL-3.0 | Strong Copyleft | Requires derivative works to be GPL |
| AGPL-3.0 | Network Copyleft | Extends copyleft to network services |
| LGPL-2.1, LGPL-3.0 | Weak Copyleft | Complex linking requirements |
| CC-BY-SA | Share-alike | Incompatible with commercial use |
| Proprietary | Commercial | Licensing costs and restrictions |

## Quick Compliance Check

Before adding a dependency:

1. **Check license**: `pip show <package-name>`
2. **Run license checker**: `make license-check`
3. **Review policy**: See [full policy](../CONTRIBUTING.md#oss-component-usage-policy)
4. **When in doubt**: Open an issue for discussion

## Tools

```bash
# Check all licenses
make license-check

# Generate report
make license-report

# Update NOTICE file
make generate-notice-direct
```