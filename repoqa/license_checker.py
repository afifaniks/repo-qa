# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""SPDX-based license consistency checker."""

import argparse
import json
import re
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from license_expression import get_spdx_licensing

# Initialize SPDX licensing
licensing = get_spdx_licensing()

# Map common license names to SPDX identifiers
LICENSE_MAPPING = {
    "MIT License": "MIT",
    "MIT": "MIT",
    "Apache Software License": "Apache-2.0",
    "Apache License": "Apache-2.0",
    "Apache 2.0": "Apache-2.0",
    "Apache-2.0": "Apache-2.0",
    "BSD License": "BSD-3-Clause",
    "BSD": "BSD-3-Clause",
    "BSD-2-Clause": "BSD-2-Clause",
    "BSD-3-Clause": "BSD-3-Clause",
    "GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "GNU General Public License v2 (GPLv2)": "GPL-2.0-only",
    "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "GNU Lesser General Public License v2.1 (LGPLv2.1)": "LGPL-2.1-only",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "ISC License (ISCL)": "ISC",
    "The Unlicense (Unlicense)": "Unlicense",
    "Python Software Foundation License": "PSF-2.0",
}


def get_requirements_packages(project_path: str = ".") -> set:
    """Get package names from requirements.txt."""
    requirements_path = Path(project_path) / "requirements.txt"
    direct_deps = set()

    if requirements_path.exists():
        requirements_content = requirements_path.read_text()
        for line in requirements_content.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                # Extract package name (remove version constraints)
                pkg_name = re.split(r"[>=<!=]", line)[0].strip()
                direct_deps.add(pkg_name.lower())

    return direct_deps


def get_dependency_licenses(
    ignore_packages: Optional[List[str]] = None,
    project_path: str = ".",
    direct_only: bool = True,
) -> List[Dict]:
    """Get dependency licenses using pip-licenses."""
    cmd = [
        "pip-licenses",
        "--format=json",
    ]

    if ignore_packages:
        for pkg in ignore_packages:
            cmd.extend(["--ignore-package", pkg])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    all_packages = json.loads(result.stdout)

    if direct_only:
        # Filter to only include packages from requirements.txt
        direct_deps = get_requirements_packages(project_path)
        filtered_packages = []

        for pkg in all_packages:
            pkg_name = pkg["Name"].lower()
            if pkg_name in direct_deps:
                filtered_packages.append(pkg)

        return filtered_packages

    return all_packages


def detect_project_license(project_path: str = ".") -> Optional[str]:
    """Detect the project's declared license as an SPDX identifier if possible."""
    project_root = Path(project_path)

    # Check pyproject.toml (PEP 621 "license" or classifiers)
    pyproject_file = project_root / "pyproject.toml"
    if pyproject_file.exists():
        try:
            with open(pyproject_file, "rb") as f:
                data = tomllib.load(f)
                project_data = data.get("project", {})

                # license field (may be string or {text=...})
                license_field = project_data.get("license")
                if isinstance(license_field, str):
                    return license_field
                elif isinstance(license_field, dict) and "text" in license_field:
                    return license_field["text"]

                # classifiers
                for classifier in project_data.get("classifiers", []):
                    if classifier.startswith("License ::"):
                        parts = classifier.split(" :: ")
                        if len(parts) >= 3:
                            return parts[-1]
        except (OSError, tomllib.TOMLDecodeError):
            pass

    # Check LICENSE files (fallback heuristic)
    for license_file in ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"]:
        license_path = project_root / license_file
        if license_path.exists():
            try:
                content = license_path.read_text(encoding="utf-8").strip().lower()
                if "mit license" in content:
                    return "MIT"
                elif "apache license" in content and "2.0" in content:
                    return "Apache-2.0"
                elif "gnu general public license" in content:
                    if "version 3" in content:
                        return "GPL-3.0-only"
                    elif "version 2" in content:
                        return "GPL-2.0-only"
                elif "bsd" in content:
                    return "BSD-2-Clause"
            except OSError:
                pass

    return None


def map_to_spdx(license_name: str) -> Optional[str]:
    """Map a license name to its SPDX identifier."""
    # Direct mapping first
    if license_name in LICENSE_MAPPING:
        return LICENSE_MAPPING[license_name]

    # Try to clean up and match common variations
    cleaned = re.sub(r"\s+", " ", license_name.strip())
    if cleaned in LICENSE_MAPPING:
        return LICENSE_MAPPING[cleaned]

    # Handle compound licenses (e.g., "MIT License; Apache 2.0")
    if ";" in license_name:
        parts = [part.strip() for part in license_name.split(";")]
        spdx_parts = []
        for part in parts:
            spdx_id = map_to_spdx(part)
            if spdx_id:
                spdx_parts.append(spdx_id)
        if spdx_parts:
            # Return as OR expression for compound licenses
            return " OR ".join(spdx_parts)

    # Try parsing directly with license-expression
    try:
        parsed = licensing.parse(license_name)
        key = parsed.key
        # Only return the key if it's in the known symbols (valid SPDX)
        if key in licensing.known_symbols:
            return key
    except Exception:
        pass

    return None


def is_permissive_license(spdx_id: str) -> bool:
    """Check if an SPDX license is permissive."""
    permissive = {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Unlicense",
        "0BSD",
    }
    return spdx_id in permissive


def is_copyleft_license(spdx_id: str) -> bool:
    """Check if an SPDX license is copyleft."""
    copyleft = {
        "GPL-2.0-only",
        "GPL-2.0-or-later",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "AGPL-3.0-only",
        "AGPL-3.0-or-later",
    }
    return spdx_id in copyleft


def is_weak_copyleft_license(spdx_id: str) -> bool:
    """Check if an SPDX license is weak copyleft."""
    weak_copyleft = {
        "LGPL-2.1-only",
        "LGPL-2.1-or-later",
        "LGPL-3.0-only",
        "LGPL-3.0-or-later",
        "MPL-2.0",
        "EPL-2.0",
        "CDDL-1.0",
    }
    return spdx_id in weak_copyleft


def check_license_compatibility(
    project_license: str, dependency_licenses: List[str]
) -> Dict[str, Any]:
    """Check license compatibility using SPDX identifiers."""
    project_spdx = map_to_spdx(project_license)

    if not project_spdx:
        return {"error": f"Could not map project license to SPDX: {project_license}"}

    compatible = []
    incompatible = []
    unknown_permissive = []

    for dep_license in dependency_licenses:
        dep_spdx = map_to_spdx(dep_license)

        if not dep_spdx:
            # Treat UNKNOWN licenses as permissive but flag them for review
            if dep_license == "UNKNOWN":
                unknown_permissive.append(dep_license)
            else:
                unknown_permissive.append(dep_license)
            continue

        # License compatibility logic
        if project_spdx == dep_spdx:
            # Same license is always compatible
            compatible.append(dep_license)
        elif is_permissive_license(dep_spdx):
            # Permissive dependencies are generally compatible
            compatible.append(dep_license)
        elif is_weak_copyleft_license(dep_spdx):
            # Weak copyleft is compatible for usage (not modification)
            compatible.append(dep_license)
        elif is_copyleft_license(dep_spdx):
            # Strong copyleft is only compatible with copyleft projects
            if is_copyleft_license(project_spdx):
                compatible.append(dep_license)
            else:
                incompatible.append(dep_license)
        else:
            # Unknown license type - be conservative
            incompatible.append(dep_license)

    result = {"compatible": compatible, "incompatible": incompatible}
    if unknown_permissive:
        result["unknown_permissive"] = unknown_permissive

    return result


def check_consistency(
    project_path: str = ".", ignore_packages: Optional[List[str]] = None
) -> Dict:
    """Check license consistency between project and dependencies."""
    project_license = detect_project_license(project_path)
    dependency_data = get_dependency_licenses(
        ignore_packages=ignore_packages, project_path=project_path, direct_only=True
    )

    dependency_licenses = [pkg["License"] for pkg in dependency_data]
    unique_dep_licenses = list(set(dependency_licenses))

    result = {
        "project_license": project_license,
        "dependency_count": len(dependency_data),
        "unique_dependency_licenses": unique_dep_licenses,
        "license_counts": {},
        "compatibility": {},
    }

    # Count occurrences of each license
    for license_name in dependency_licenses:
        result["license_counts"][license_name] = (
            result["license_counts"].get(license_name, 0) + 1
        )

    # Check compatibility if project license is detected
    if project_license:
        result["compatibility"] = check_license_compatibility(
            project_license, unique_dep_licenses
        )
    else:
        result["compatibility"] = {"error": "Could not detect project license"}

    return result


def print_report(result: Dict) -> None:
    """Print a human-readable license consistency report."""
    print("=" * 60)
    print("LICENSE CONSISTENCY REPORT")
    print("=" * 60)

    print(f"\n📋 Project License: {result.get('project_license', 'Unknown')}")
    print(f"📦 Total Dependencies: {result.get('dependency_count', 0)}")
    unique_licenses = len(result.get("unique_dependency_licenses", []))
    print(f"📄 Unique Licenses: {unique_licenses}")

    # License distribution
    print("\n📊 License Distribution:")
    license_counts = result.get("license_counts", {})
    for license_name, count in sorted(
        license_counts.items(), key=lambda x: x[1], reverse=True
    ):
        display_name = (
            license_name[:47] + "..." if len(license_name) > 50 else license_name
        )
        print(f"   {count:3d} packages: {display_name}")

    # Compatibility analysis
    compatibility = result.get("compatibility", {})

    if "error" in compatibility:
        print(f"\n❌ {compatibility['error']}")
        return

    compatible = compatibility.get("compatible", [])
    incompatible = compatibility.get("incompatible", [])
    unknown_permissive = compatibility.get("unknown_permissive", [])

    print(f"\n✅ Compatible Licenses ({len(compatible)}):")
    for license_name in sorted(compatible):
        spdx_id = map_to_spdx(license_name)
        display_name = (
            license_name[:47] + "..." if len(license_name) > 50 else license_name
        )
        spdx_suffix = f" → {spdx_id}" if spdx_id else ""
        print(f"   ✓ {display_name}{spdx_suffix}")

    if unknown_permissive:
        print(
            f"\n⚠️  Unknown Licenses (Treated as Permissive) "
            f"({len(unknown_permissive)}):"
        )
        for license_name in sorted(unknown_permissive):
            display_name = (
                license_name[:47] + "..." if len(license_name) > 50 else license_name
            )
            print(f"   ⚠ {display_name} (assumed permissive)")

    if incompatible:
        print(f"\n❌ Incompatible Licenses ({len(incompatible)}):")
        for license_name in sorted(incompatible):
            spdx_id = map_to_spdx(license_name)
            display_name = (
                license_name[:47] + "..." if len(license_name) > 50 else license_name
            )
            spdx_suffix = f" → {spdx_id}" if spdx_id else ""
            print(f"   ❌ {display_name}{spdx_suffix}")

    if incompatible or unknown_permissive:
        print("\n💡 Recommendations:")
        if incompatible:
            print("   • Review incompatible licenses - may need replacement")
        if unknown_permissive:
            print("   • Verify unknown licenses are actually permissive")
            print("   • Map unknown licenses to SPDX identifiers")
        print("   • Consult with legal team for commercial projects")
    else:
        print("\n🎉 All dependency licenses are compatible!")

    print("\n" + "=" * 60)


def generate_notice_file(
    project_path: str = ".",
    ignore_packages: Optional[List[str]] = None,
    project_name: str = "repo-qa",
    project_copyright: str = "Copyright (c) 2025 Afif Al Mamun",
    direct_only: bool = False,
) -> str:
    """Generate a NOTICE file content for third-party attributions."""
    dependencies = get_dependency_licenses(ignore_packages)

    # Get direct dependencies if requested
    direct_deps = set()
    if direct_only:
        requirements_path = Path(project_path) / "requirements.txt"
        if requirements_path.exists():
            requirements_content = requirements_path.read_text()
            for line in requirements_content.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    # Extract package name (remove version constraints)
                    pkg_name = re.split(r"[>=<!=]", line)[0].strip()
                    direct_deps.add(pkg_name.lower())

    # Get detailed package info with authors
    cmd = ["pip-licenses", "--format=json", "--with-authors"]
    if ignore_packages:
        for pkg in ignore_packages:
            cmd.extend(["--ignore-package", pkg])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    detailed_deps = json.loads(result.stdout)

    # Create lookup dict for detailed info
    dep_details = {dep["Name"]: dep for dep in detailed_deps}

    # Group by license type
    apache_deps = []
    bsd_deps = []
    mit_deps = []
    mozilla_deps = []
    dual_license_deps = []
    other_deps = []

    for dep in dependencies:
        name = dep["Name"]
        license_name = dep["License"]

        details = dep_details.get(name, {})
        author = details.get("Author", "")

        dep_info = {
            "name": name,
            "license": license_name,
            "author": author,
            "spdx": map_to_spdx(license_name),
        }

        if "apache" in license_name.lower():
            apache_deps.append(dep_info)
        elif "bsd" in license_name.lower():
            bsd_deps.append(dep_info)
        elif license_name == "MIT" or license_name == "MIT License":
            mit_deps.append(dep_info)
        elif "mozilla" in license_name.lower() or "mpl" in license_name.lower():
            mozilla_deps.append(dep_info)
        elif ";" in license_name:  # Dual license
            dual_license_deps.append(dep_info)
        else:
            other_deps.append(dep_info)

    # Generate NOTICE content
    notice_content = f"""NOTICE

{project_name}
{project_copyright}

This software contains code from the following third-party libraries and projects:

================================================================================

APACHE SOFTWARE LICENSE 2.0 COMPONENTS

The following components are provided under the Apache Software License 2.0:

"""

    for dep in sorted(apache_deps, key=lambda x: x["name"]):
        notice_content += f"""{dep["name"]}
"""
        if dep["author"] and dep["author"] != "UNKNOWN":
            notice_content += f"""Copyright (c) {dep["author"]}
"""
        notice_content += f"""License: {dep["license"]}

"""

    notice_content += """================================================================================

BSD LICENSE COMPONENTS

The following components are provided under various BSD licenses:

"""

    for dep in sorted(bsd_deps, key=lambda x: x["name"]):
        notice_content += f"""{dep["name"]}
"""
        if dep["author"] and dep["author"] != "UNKNOWN":
            notice_content += f"""Copyright (c) {dep["author"]}
"""
        notice_content += f"""License: {dep["license"]}

"""

    if mit_deps:
        notice_content += """================================================================================

MIT LICENSE COMPONENTS

The following components are provided under the MIT License:

"""
        for dep in sorted(mit_deps, key=lambda x: x["name"]):
            notice_content += f"""{dep["name"]}
"""
            if dep["author"] and dep["author"] != "UNKNOWN":
                notice_content += f"""Copyright (c) {dep["author"]}
"""
            notice_content += f"""License: {dep["license"]}

"""

    if mozilla_deps:
        notice_content += """================================================================================

MOZILLA PUBLIC LICENSE 2.0 COMPONENTS

The following components are provided under the Mozilla Public License 2.0:

"""
        for dep in sorted(mozilla_deps, key=lambda x: x["name"]):
            notice_content += f"""{dep["name"]}
"""
            if dep["author"] and dep["author"] != "UNKNOWN":
                notice_content += f"""Copyright (c) {dep["author"]}
"""
            notice_content += f"""License: {dep["license"]}

"""

    if dual_license_deps:
        notice_content += """================================================================================

DUAL-LICENSE COMPONENTS

The following components are provided under multiple licenses:

"""
        for dep in sorted(dual_license_deps, key=lambda x: x["name"]):
            notice_content += f"""{dep["name"]}
"""
            if dep["author"] and dep["author"] != "UNKNOWN":
                notice_content += f"""Copyright (c) {dep["author"]}
"""
            notice_content += f"""License: {dep["license"]}

"""

    if other_deps:
        notice_content += """================================================================================

OTHER LICENSES

"""
        for dep in sorted(other_deps, key=lambda x: x["name"]):
            notice_content += f"""{dep["name"]}
"""
            if dep["author"] and dep["author"] != "UNKNOWN":
                notice_content += f"""Copyright (c) {dep["author"]}
"""
            notice_content += f"""License: {dep["license"]}

"""

    notice_content += f"""================================================================================

This NOTICE file is provided for compliance with the license terms of the
included third-party software. The original license texts for each component
can be found in their respective package directories or at their project
repositories.

For questions about licensing, please contact: afifaniks@gmail.com

Last updated: {datetime.now().strftime("%B %d, %Y")}"""

    return notice_content


def main():
    parser = argparse.ArgumentParser(description="Check license consistency")
    parser.add_argument(
        "--format", choices=["json", "report"], default="report", help="Output format"
    )
    parser.add_argument("--ignore", type=str, nargs="+", help="Packages to ignore")
    parser.add_argument(
        "--project-path", type=str, default=".", help="Path to project root"
    )
    parser.add_argument(
        "--generate-notice",
        action="store_true",
        help="Generate NOTICE file for attribution",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Include only direct dependencies (from requirements.txt)",
    )

    args = parser.parse_args()

    if args.generate_notice:
        notice_content = generate_notice_file(
            project_path=args.project_path,
            ignore_packages=args.ignore,
            direct_only=args.direct_only,
        )
        notice_path = Path(args.project_path) / "NOTICE"
        notice_path.write_text(notice_content)
        print(f"Generated NOTICE file at: {notice_path}")
        return

    result = check_consistency(
        project_path=args.project_path, ignore_packages=args.ignore
    )

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
