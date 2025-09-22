"""
Fixture loading utilities for MathIR Parser testing.

This module provides utilities for loading and managing test fixtures,
including mathematical expressions, expected results, and test scenarios.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator
from dataclasses import dataclass, field
import yaml


@dataclass
class TestFixture:
    """Data class representing a test fixture."""
    name: str
    description: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_file: Optional[str] = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the fixture data."""
        return self.data.get(key, default)
    
    def has(self, key: str) -> bool:
        """Check if fixture has a specific key."""
        return key in self.data


class FixtureLoader:
    """Loader for test fixtures from various sources."""
    
    def __init__(self, fixtures_root: Union[str, Path]):
        """
        Initialize fixture loader.
        
        Args:
            fixtures_root: Root directory containing fixture files
        """
        self.fixtures_root = Path(fixtures_root)
        self._cache: Dict[str, TestFixture] = {}
        self._loaded_files: Dict[str, List[TestFixture]] = {}
    
    def load_fixture_file(self, filepath: Union[str, Path]) -> List[TestFixture]:
        """
        Load fixtures from a single file.
        
        Args:
            filepath: Path to fixture file
            
        Returns:
            List of TestFixture objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported or invalid
        """
        filepath = Path(filepath)
        
        if not filepath.is_absolute():
            filepath = self.fixtures_root / filepath
        
        if not filepath.exists():
            raise FileNotFoundError(f"Fixture file not found: {filepath}")
        
        # Check cache
        cache_key = str(filepath)
        if cache_key in self._loaded_files:
            return self._loaded_files[cache_key]
        
        # Load based on file extension
        if filepath.suffix.lower() == '.json':
            fixtures = self._load_json_fixtures(filepath)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            fixtures = self._load_yaml_fixtures(filepath)
        else:
            raise ValueError(f"Unsupported fixture file format: {filepath.suffix}")
        
        # Cache the loaded fixtures
        self._loaded_files[cache_key] = fixtures
        
        # Add to individual fixture cache
        for fixture in fixtures:
            self._cache[fixture.name] = fixture
        
        return fixtures
    
    def _load_json_fixtures(self, filepath: Path) -> List[TestFixture]:
        """Load fixtures from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fixtures = []
        
        if isinstance(data, list):
            # Array of fixtures
            for i, item in enumerate(data):
                fixture = self._create_fixture_from_dict(item, f"{filepath.stem}_{i}", str(filepath))
                fixtures.append(fixture)
        elif isinstance(data, dict):
            if 'fixtures' in data:
                # Structured format with fixtures array
                for i, item in enumerate(data['fixtures']):
                    fixture = self._create_fixture_from_dict(item, f"{filepath.stem}_{i}", str(filepath))
                    fixtures.append(fixture)
            else:
                # Single fixture
                fixture = self._create_fixture_from_dict(data, filepath.stem, str(filepath))
                fixtures.append(fixture)
        
        return fixtures
    
    def _load_yaml_fixtures(self, filepath: Path) -> List[TestFixture]:
        """Load fixtures from YAML file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        fixtures = []
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                fixture = self._create_fixture_from_dict(item, f"{filepath.stem}_{i}", str(filepath))
                fixtures.append(fixture)
        elif isinstance(data, dict):
            if 'fixtures' in data:
                for i, item in enumerate(data['fixtures']):
                    fixture = self._create_fixture_from_dict(item, f"{filepath.stem}_{i}", str(filepath))
                    fixtures.append(fixture)
            else:
                fixture = self._create_fixture_from_dict(data, filepath.stem, str(filepath))
                fixtures.append(fixture)
        
        return fixtures
    
    def _create_fixture_from_dict(self, data: Dict[str, Any], default_name: str, source_file: str) -> TestFixture:
        """Create TestFixture from dictionary data."""
        name = data.get('name', default_name)
        description = data.get('description', f"Fixture from {source_file}")
        metadata = data.get('metadata', {})
        
        # Remove metadata fields from main data
        fixture_data = {k: v for k, v in data.items() 
                       if k not in ['name', 'description', 'metadata']}
        
        return TestFixture(
            name=name,
            description=description,
            data=fixture_data,
            metadata=metadata,
            source_file=source_file
        )
    
    def load_fixtures_from_directory(self, directory: Union[str, Path], 
                                   pattern: str = "*.json") -> List[TestFixture]:
        """
        Load all fixtures from a directory matching a pattern.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            
        Returns:
            List of all loaded fixtures
        """
        directory = Path(directory)
        if not directory.is_absolute():
            directory = self.fixtures_root / directory
        
        fixtures = []
        for filepath in directory.glob(pattern):
            try:
                file_fixtures = self.load_fixture_file(filepath)
                fixtures.extend(file_fixtures)
            except Exception as e:
                print(f"Warning: Failed to load fixtures from {filepath}: {e}")
        
        return fixtures
    
    def get_fixture(self, name: str) -> Optional[TestFixture]:
        """
        Get a fixture by name.
        
        Args:
            name: Fixture name
            
        Returns:
            TestFixture if found, None otherwise
        """
        return self._cache.get(name)
    
    def get_fixtures_by_category(self, category: str) -> List[TestFixture]:
        """
        Get all fixtures belonging to a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of fixtures in the category
        """
        fixtures = []
        for fixture in self._cache.values():
            if fixture.metadata.get('category') == category:
                fixtures.append(fixture)
        return fixtures
    
    def get_fixtures_by_tag(self, tag: str) -> List[TestFixture]:
        """
        Get all fixtures with a specific tag.
        
        Args:
            tag: Tag name
            
        Returns:
            List of fixtures with the tag
        """
        fixtures = []
        for fixture in self._cache.values():
            tags = fixture.metadata.get('tags', [])
            if tag in tags:
                fixtures.append(fixture)
        return fixtures
    
    def list_all_fixtures(self) -> List[TestFixture]:
        """Get all loaded fixtures."""
        return list(self._cache.values())
    
    def clear_cache(self):
        """Clear the fixture cache."""
        self._cache.clear()
        self._loaded_files.clear()


class FixtureManager:
    """Manager for organizing and accessing test fixtures."""
    
    def __init__(self, fixtures_root: Union[str, Path]):
        """
        Initialize fixture manager.
        
        Args:
            fixtures_root: Root directory containing fixture files
        """
        self.loader = FixtureLoader(fixtures_root)
        self.fixtures_root = Path(fixtures_root)
        self._auto_loaded = False
    
    def auto_load_fixtures(self):
        """Automatically load all fixtures from standard directories."""
        if self._auto_loaded:
            return
        
        standard_dirs = [
            "mathematical_expressions",
            "edge_cases",
            "expected_results",
            "integration_scenarios",
            "performance_benchmarks"
        ]
        
        for dir_name in standard_dirs:
            dir_path = self.fixtures_root / dir_name
            if dir_path.exists():
                try:
                    self.loader.load_fixtures_from_directory(dir_path)
                except Exception as e:
                    print(f"Warning: Failed to auto-load fixtures from {dir_path}: {e}")
        
        self._auto_loaded = True
    
    def get_mathematical_expressions(self, category: Optional[str] = None) -> List[TestFixture]:
        """
        Get mathematical expression fixtures.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of mathematical expression fixtures
        """
        self.auto_load_fixtures()
        
        fixtures = self.loader.get_fixtures_by_category("mathematical_expressions")
        
        if category:
            fixtures = [f for f in fixtures if f.metadata.get('subcategory') == category]
        
        return fixtures
    
    def get_edge_cases(self) -> List[TestFixture]:
        """Get edge case fixtures."""
        self.auto_load_fixtures()
        return self.loader.get_fixtures_by_category("edge_cases")
    
    def get_expected_results(self, test_type: Optional[str] = None) -> List[TestFixture]:
        """
        Get expected result fixtures.
        
        Args:
            test_type: Optional test type filter
            
        Returns:
            List of expected result fixtures
        """
        self.auto_load_fixtures()
        
        fixtures = self.loader.get_fixtures_by_category("expected_results")
        
        if test_type:
            fixtures = [f for f in fixtures if f.metadata.get('test_type') == test_type]
        
        return fixtures
    
    def get_integration_scenarios(self) -> List[TestFixture]:
        """Get integration test scenario fixtures."""
        self.auto_load_fixtures()
        return self.loader.get_fixtures_by_category("integration_scenarios")
    
    def get_performance_benchmarks(self) -> List[TestFixture]:
        """Get performance benchmark fixtures."""
        self.auto_load_fixtures()
        return self.loader.get_fixtures_by_category("performance_benchmarks")
    
    def get_fixtures_for_target_type(self, target_type: str) -> List[TestFixture]:
        """
        Get fixtures for a specific target type.
        
        Args:
            target_type: Target type (e.g., 'integral_def', 'limit', 'sum')
            
        Returns:
            List of fixtures for the target type
        """
        self.auto_load_fixtures()
        
        fixtures = []
        for fixture in self.loader.list_all_fixtures():
            # Check if fixture has mathir data with matching target type
            mathir_data = fixture.get('mathir')
            if mathir_data and 'targets' in mathir_data:
                for target in mathir_data['targets']:
                    if target.get('type') == target_type:
                        fixtures.append(fixture)
                        break
        
        return fixtures
    
    def create_fixture_iterator(self, 
                              category: Optional[str] = None,
                              tags: Optional[List[str]] = None) -> Iterator[TestFixture]:
        """
        Create an iterator over fixtures with optional filtering.
        
        Args:
            category: Optional category filter
            tags: Optional list of tags to filter by
            
        Yields:
            TestFixture objects matching the criteria
        """
        self.auto_load_fixtures()
        
        for fixture in self.loader.list_all_fixtures():
            # Category filter
            if category and fixture.metadata.get('category') != category:
                continue
            
            # Tags filter
            if tags:
                fixture_tags = fixture.metadata.get('tags', [])
                if not any(tag in fixture_tags for tag in tags):
                    continue
            
            yield fixture
    
    def validate_fixture_structure(self, fixture: TestFixture, 
                                 required_fields: List[str]) -> bool:
        """
        Validate that a fixture has required fields.
        
        Args:
            fixture: Fixture to validate
            required_fields: List of required field names
            
        Returns:
            True if fixture has all required fields
        """
        for field in required_fields:
            if not fixture.has(field):
                return False
        return True
    
    def get_fixture_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded fixtures."""
        self.auto_load_fixtures()
        
        fixtures = self.loader.list_all_fixtures()
        
        categories = {}
        tags = {}
        
        for fixture in fixtures:
            # Count categories
            category = fixture.metadata.get('category', 'uncategorized')
            categories[category] = categories.get(category, 0) + 1
            
            # Count tags
            fixture_tags = fixture.metadata.get('tags', [])
            for tag in fixture_tags:
                tags[tag] = tags.get(tag, 0) + 1
        
        return {
            "total_fixtures": len(fixtures),
            "categories": categories,
            "tags": tags,
            "source_files": len(self.loader._loaded_files)
        }


# Convenience functions
def load_fixture_file(filepath: Union[str, Path], 
                     fixtures_root: Optional[Union[str, Path]] = None) -> List[TestFixture]:
    """
    Convenience function to load fixtures from a file.
    
    Args:
        filepath: Path to fixture file
        fixtures_root: Root directory (uses file's parent if None)
        
    Returns:
        List of TestFixture objects
    """
    if fixtures_root is None:
        fixtures_root = Path(filepath).parent
    
    loader = FixtureLoader(fixtures_root)
    return loader.load_fixture_file(filepath)


def create_fixture_from_dict(name: str, description: str, data: Dict[str, Any], 
                           metadata: Optional[Dict[str, Any]] = None) -> TestFixture:
    """
    Convenience function to create a fixture from dictionary data.
    
    Args:
        name: Fixture name
        description: Fixture description
        data: Fixture data
        metadata: Optional metadata
        
    Returns:
        TestFixture object
    """
    return TestFixture(
        name=name,
        description=description,
        data=data,
        metadata=metadata or {}
    )