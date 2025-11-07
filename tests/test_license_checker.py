# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Tests for license checker module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest


class TestLicenseMapping:
    """Test suite for SPDX license mapping."""

    def test_map_to_spdx_direct_mapping(self):
        """Test direct license name to SPDX mapping."""
        from repoqa.license_checker import map_to_spdx

        assert map_to_spdx("MIT") == "MIT"
        assert map_to_spdx("MIT License") == "MIT"
        assert map_to_spdx("Apache-2.0") == "Apache-2.0"
        assert map_to_spdx("Apache Software License") == "Apache-2.0"
        assert map_to_spdx("BSD License") == "BSD-3-Clause"

    def test_map_to_spdx_with_whitespace(self):
        """Test mapping with extra whitespace."""
        from repoqa.license_checker import map_to_spdx

        assert map_to_spdx("  MIT  ") == "MIT"
        assert map_to_spdx("Apache  2.0") == "Apache-2.0"

    def test_map_to_spdx_compound_licenses(self):
        """Test mapping compound licenses (OR expressions)."""
        from repoqa.license_checker import map_to_spdx

        result = map_to_spdx("MIT; Apache-2.0")
        assert "MIT" in result
        assert "Apache-2.0" in result
        assert " OR " in result

    def test_map_to_spdx_unknown_license(self):
        """Test mapping unknown license returns None."""
        from repoqa.license_checker import map_to_spdx

        assert map_to_spdx("Unknown Custom License") is None
        assert map_to_spdx("") is None

    def test_map_to_spdx_gpl_licenses(self):
        """Test GPL license mappings."""
        from repoqa.license_checker import map_to_spdx

        assert map_to_spdx("GNU General Public License v3 (GPLv3)") == "GPL-3.0-only"
        assert map_to_spdx("GNU General Public License v2 (GPLv2)") == "GPL-2.0-only"


class TestLicenseClassification:
    """Test suite for license classification."""

    def test_is_permissive_license(self):
        """Test permissive license detection."""
        from repoqa.license_checker import is_permissive_license

        assert is_permissive_license("MIT") is True
        assert is_permissive_license("Apache-2.0") is True
        assert is_permissive_license("BSD-2-Clause") is True
        assert is_permissive_license("BSD-3-Clause") is True
        assert is_permissive_license("ISC") is True
        assert is_permissive_license("Unlicense") is True

    def test_is_permissive_license_negative(self):
        """Test non-permissive licenses."""
        from repoqa.license_checker import is_permissive_license

        assert is_permissive_license("GPL-3.0-only") is False
        assert is_permissive_license("LGPL-3.0-only") is False
        assert is_permissive_license("AGPL-3.0-only") is False

    def test_is_copyleft_license(self):
        """Test copyleft license detection."""
        from repoqa.license_checker import is_copyleft_license

        assert is_copyleft_license("GPL-2.0-only") is True
        assert is_copyleft_license("GPL-3.0-only") is True
        assert is_copyleft_license("AGPL-3.0-only") is True
        assert is_copyleft_license("GPL-2.0-or-later") is True

    def test_is_copyleft_license_negative(self):
        """Test non-copyleft licenses."""
        from repoqa.license_checker import is_copyleft_license

        assert is_copyleft_license("MIT") is False
        assert is_copyleft_license("Apache-2.0") is False
        assert is_copyleft_license("LGPL-3.0-only") is False

    def test_is_weak_copyleft_license(self):
        """Test weak copyleft license detection."""
        from repoqa.license_checker import is_weak_copyleft_license

        assert is_weak_copyleft_license("LGPL-2.1-only") is True
        assert is_weak_copyleft_license("LGPL-3.0-only") is True
        assert is_weak_copyleft_license("MPL-2.0") is True
        assert is_weak_copyleft_license("EPL-2.0") is True

    def test_is_weak_copyleft_license_negative(self):
        """Test non-weak-copyleft licenses."""
        from repoqa.license_checker import is_weak_copyleft_license

        assert is_weak_copyleft_license("MIT") is False
        assert is_weak_copyleft_license("GPL-3.0-only") is False


class TestProjectLicenseDetection:
    """Test suite for project license detection."""

    @patch("pathlib.Path.exists")
    @patch("repoqa.license_checker.open", new_callable=mock_open)
    @patch("tomllib.load")
    def test_detect_project_license_from_pyproject_toml(
        self, mock_tomllib_load, mock_file, mock_exists
    ):
        """Test detecting license from pyproject.toml."""
        from repoqa.license_checker import detect_project_license

        # Mock pyproject.toml exists
        def exists_side_effect():
            # This will be called on the mock Path object
            # We need to check which Path is calling exists()
            # Since mock_exists is bound to Path.exists, we can't access self
            # So we just return True for pyproject.toml case
            return True

        mock_exists.side_effect = exists_side_effect

        # Mock tomllib.load to return project data
        mock_tomllib_load.return_value = {"project": {"license": "MIT"}}

        result = detect_project_license(".")

        assert result == "MIT"

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    def test_detect_project_license_from_license_file(self, mock_read, mock_exists):
        """Test detecting license from LICENSE file."""
        from repoqa.license_checker import detect_project_license

        # Mock LICENSE file exists, pyproject.toml doesn't
        # We'll use a call count to differentiate between calls
        call_count = {"count": 0}

        def exists_side_effect():
            call_count["count"] += 1
            # First call is pyproject.toml (should be False)
            # Second call is LICENSE (should be True)
            return call_count["count"] > 1

        mock_exists.side_effect = exists_side_effect

        # Mock MIT license content
        mock_read.return_value = """
MIT License

Copyright (c) 2025 Test

Permission is hereby granted...
"""

        result = detect_project_license(".")
        assert result == "MIT"

    @patch("pathlib.Path.exists")
    def test_detect_project_license_not_found(self, mock_exists):
        """Test when project license cannot be detected."""
        from repoqa.license_checker import detect_project_license

        mock_exists.return_value = False
        result = detect_project_license(".")
        assert result is None


class TestGetRequirementsPackages:
    """Test suite for getting packages from requirements.txt."""

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    def test_get_requirements_packages(self, mock_read, mock_exists):
        """Test parsing requirements.txt."""
        from repoqa.license_checker import get_requirements_packages

        mock_exists.return_value = True
        mock_read.return_value = """
# Comment line
requests>=2.28.0
numpy==1.24.0
pytest
# Another comment
pandas>=1.5.0,<2.0.0
"""

        result = get_requirements_packages(".")
        assert "requests" in result
        assert "numpy" in result
        assert "pytest" in result
        assert "pandas" in result
        assert len(result) == 4

    @patch("pathlib.Path.exists")
    def test_get_requirements_packages_no_file(self, mock_exists):
        """Test when requirements.txt doesn't exist."""
        from repoqa.license_checker import get_requirements_packages

        mock_exists.return_value = False
        result = get_requirements_packages(".")
        assert result == set()


class TestLicenseCompatibility:
    """Test suite for license compatibility checking."""

    def test_check_license_compatibility_mit_project(self):
        """Test compatibility check for MIT project."""
        from repoqa.license_checker import check_license_compatibility

        result = check_license_compatibility(
            "MIT", ["MIT", "Apache-2.0", "BSD-3-Clause", "LGPL-3.0-only"]
        )

        assert "compatible" in result
        assert "MIT" in result["compatible"]
        assert "Apache-2.0" in result["compatible"]
        assert "BSD-3-Clause" in result["compatible"]
        assert "LGPL-3.0-only" in result["compatible"]

    def test_check_license_compatibility_mit_with_gpl(self):
        """Test MIT project with GPL dependency (incompatible)."""
        from repoqa.license_checker import check_license_compatibility

        result = check_license_compatibility(
            "MIT",
            [
                "GPL-3.0-only",
            ],
        )

        assert "incompatible" in result
        assert "GPL-3.0-only" in result["incompatible"]

    def test_check_license_compatibility_unknown_license(self):
        """Test handling unknown licenses."""
        from repoqa.license_checker import check_license_compatibility

        result = check_license_compatibility("MIT", ["UNKNOWN"])

        assert "unknown_permissive" in result
        assert "UNKNOWN" in result["unknown_permissive"]

    def test_check_license_compatibility_gpl_project(self):
        """Test GPL project accepts GPL dependencies."""
        from repoqa.license_checker import check_license_compatibility

        result = check_license_compatibility(
            "GPL-3.0-only", ["GPL-3.0-only", "LGPL-3.0-only", "MIT"]
        )

        assert "compatible" in result
        assert "GPL-3.0-only" in result["compatible"]
        assert "LGPL-3.0-only" in result["compatible"]
        assert "MIT" in result["compatible"]

    def test_check_license_compatibility_invalid_project_license(self):
        """Test with invalid project license."""
        from repoqa.license_checker import check_license_compatibility

        result = check_license_compatibility("Invalid License Name", ["MIT"])

        assert "error" in result


class TestGetDependencyLicenses:
    """Test suite for getting dependency licenses."""

    @patch("subprocess.run")
    def test_get_dependency_licenses(self, mock_run):
        """Test getting dependency licenses from pip-licenses."""
        from repoqa.license_checker import get_dependency_licenses

        # Mock subprocess output
        mock_result = Mock()
        mock_result.stdout = json.dumps(
            [
                {"Name": "requests", "License": "Apache-2.0"},
                {"Name": "numpy", "License": "BSD-3-Clause"},
                {"Name": "pytest", "License": "MIT"},
            ]
        )
        mock_run.return_value = mock_result

        result = get_dependency_licenses(direct_only=False)

        assert len(result) == 3
        assert result[0]["Name"] == "requests"
        assert result[0]["License"] == "Apache-2.0"

    @patch("subprocess.run")
    @patch("repoqa.license_checker.get_requirements_packages")
    def test_get_dependency_licenses_direct_only(self, mock_get_reqs, mock_run):
        """Test filtering to direct dependencies only."""
        from repoqa.license_checker import get_dependency_licenses

        # Mock requirements.txt packages
        mock_get_reqs.return_value = {"requests", "numpy"}

        # Mock pip-licenses output with more packages
        mock_result = Mock()
        mock_result.stdout = json.dumps(
            [
                {"Name": "requests", "License": "Apache-2.0"},
                {"Name": "numpy", "License": "BSD-3-Clause"},
                {"Name": "urllib3", "License": "MIT"},  # transitive dep
                {"Name": "certifi", "License": "MPL-2.0"},  # transitive
            ]
        )
        mock_run.return_value = mock_result

        result = get_dependency_licenses(direct_only=True)

        # Should only include requests and numpy
        assert len(result) == 2
        names = [dep["Name"] for dep in result]
        assert "requests" in names
        assert "numpy" in names
        assert "urllib3" not in names
        assert "certifi" not in names


class TestCheckConsistency:
    """Test suite for overall consistency checking."""

    @patch("repoqa.license_checker.detect_project_license")
    @patch("repoqa.license_checker.get_dependency_licenses")
    def test_check_consistency(self, mock_get_deps, mock_detect_license):
        """Test overall license consistency check."""
        from repoqa.license_checker import check_consistency

        mock_detect_license.return_value = "MIT"
        mock_get_deps.return_value = [
            {"Name": "requests", "License": "Apache-2.0"},
            {"Name": "numpy", "License": "BSD-3-Clause"},
            {"Name": "pytest", "License": "MIT"},
        ]

        result = check_consistency(".")

        assert result["project_license"] == "MIT"
        assert result["dependency_count"] == 3
        assert len(result["unique_dependency_licenses"]) == 3
        assert "license_counts" in result
        assert "compatibility" in result

    @patch("repoqa.license_checker.detect_project_license")
    @patch("repoqa.license_checker.get_dependency_licenses")
    def test_check_consistency_no_project_license(
        self, mock_get_deps, mock_detect_license
    ):
        """Test when project license cannot be detected."""
        from repoqa.license_checker import check_consistency

        mock_detect_license.return_value = None
        mock_get_deps.return_value = [
            {"Name": "requests", "License": "Apache-2.0"},
        ]

        result = check_consistency(".")

        assert result["project_license"] is None
        assert "error" in result["compatibility"]


class TestPrintReport:
    """Test suite for report printing."""

    @patch("builtins.print")
    def test_print_report_success(self, mock_print):
        """Test printing a successful report."""
        from repoqa.license_checker import print_report

        result = {
            "project_license": "MIT",
            "dependency_count": 5,
            "unique_dependency_licenses": ["MIT", "Apache-2.0"],
            "license_counts": {"MIT": 3, "Apache-2.0": 2},
            "compatibility": {
                "compatible": ["MIT", "Apache-2.0"],
                "incompatible": [],
            },
        }

        print_report(result)

        # Verify print was called with report content
        assert mock_print.called
        calls = [str(call) for call in mock_print.call_args_list]
        report_text = " ".join(calls)
        assert "LICENSE CONSISTENCY REPORT" in report_text
        assert "MIT" in report_text

    @patch("builtins.print")
    def test_print_report_with_incompatible(self, mock_print):
        """Test printing report with incompatible licenses."""
        from repoqa.license_checker import print_report

        result = {
            "project_license": "MIT",
            "dependency_count": 2,
            "unique_dependency_licenses": ["MIT", "GPL-3.0-only"],
            "license_counts": {"MIT": 1, "GPL-3.0-only": 1},
            "compatibility": {
                "compatible": ["MIT"],
                "incompatible": ["GPL-3.0-only"],
            },
        }

        print_report(result)

        assert mock_print.called
        calls = [str(call) for call in mock_print.call_args_list]
        report_text = " ".join(calls)
        assert "Incompatible Licenses" in report_text


class TestGenerateNoticeFile:
    """Test suite for NOTICE file generation."""

    @patch("subprocess.run")
    @patch("repoqa.license_checker.get_dependency_licenses")
    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.exists")
    def test_generate_notice_file(
        self, mock_exists, mock_read, mock_get_deps, mock_run
    ):
        """Test generating NOTICE file content."""
        from repoqa.license_checker import generate_notice_file

        mock_exists.return_value = True
        mock_read.return_value = "requests>=2.28.0\nnumpy==1.24.0"

        mock_get_deps.return_value = [
            {"Name": "requests", "License": "Apache-2.0"},
            {"Name": "numpy", "License": "BSD-3-Clause"},
        ]

        # Mock detailed output with authors
        mock_result = Mock()
        mock_result.stdout = json.dumps(
            [
                {
                    "Name": "requests",
                    "License": "Apache-2.0",
                    "Author": "Kenneth Reitz",
                },
                {
                    "Name": "numpy",
                    "License": "BSD-3-Clause",
                    "Author": "NumPy Developers",
                },
            ]
        )
        mock_run.return_value = mock_result

        result = generate_notice_file()

        assert "NOTICE" in result
        assert "repo-qa" in result
        assert "requests" in result
        assert "numpy" in result
        assert "APACHE SOFTWARE LICENSE 2.0" in result.upper()
        assert "BSD LICENSE" in result.upper()

    @patch("subprocess.run")
    @patch("repoqa.license_checker.get_dependency_licenses")
    def test_generate_notice_file_mit_licenses(self, mock_get_deps, mock_run):
        """Test generating NOTICE with MIT licensed dependencies."""
        from repoqa.license_checker import generate_notice_file

        mock_get_deps.return_value = [
            {"Name": "pytest", "License": "MIT"},
        ]

        mock_result = Mock()
        mock_result.stdout = json.dumps(
            [
                {"Name": "pytest", "License": "MIT", "Author": "Pytest Team"},
            ]
        )
        mock_run.return_value = mock_result

        result = generate_notice_file()

        assert "MIT LICENSE COMPONENTS" in result
        assert "pytest" in result


class TestMainFunction:
    """Test suite for main CLI function."""

    @patch("repoqa.license_checker.check_consistency")
    @patch("repoqa.license_checker.print_report")
    @patch("sys.argv", ["license_checker.py", "--format", "report"])
    def test_main_report_format(self, mock_print, mock_check):
        """Test main function with report format."""
        from repoqa.license_checker import main

        mock_check.return_value = {
            "project_license": "MIT",
            "dependency_count": 5,
        }

        main()

        assert mock_check.called
        assert mock_print.called

    @patch("repoqa.license_checker.check_consistency")
    @patch("builtins.print")
    @patch("sys.argv", ["license_checker.py", "--format", "json"])
    def test_main_json_format(self, mock_print, mock_check):
        """Test main function with JSON format."""
        from repoqa.license_checker import main

        mock_check.return_value = {
            "project_license": "MIT",
            "dependency_count": 5,
        }

        main()

        assert mock_check.called
        assert mock_print.called

    @patch("repoqa.license_checker.generate_notice_file")
    @patch("pathlib.Path.write_text")
    @patch("builtins.print")
    @patch("sys.argv", ["license_checker.py", "--generate-notice"])
    def test_main_generate_notice(self, mock_print, mock_write, mock_generate):
        """Test main function generating NOTICE file."""
        from repoqa.license_checker import main

        mock_generate.return_value = "NOTICE content"

        main()

        assert mock_generate.called
        assert mock_write.called
