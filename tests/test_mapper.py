"""
Unit tests for the mapper module.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.detector import Detection
from src.mapping.mapper import Mapper, MappedObject


class TestMapper(unittest.TestCase):
    """Tests for the Mapper class."""
    
    def setUp(self):
        """Set up test case."""
        # Create a mapper instance
        self.mapper = Mapper(
            persistence_file=None,  # No persistence for testing
            distance_threshold=10.0
        )
        
        # Create a list of localized objects
        self.localized_objects = [
            {
                'detection': Detection(
                    class_id=0,
                    class_name='traffic_light_red',
                    confidence=0.95,
                    bbox=(900, 400, 1020, 600),
                    state='red'
                ),
                'position': MagicMock(
                    latitude=37.7749,
                    longitude=-122.4194,
                    altitude=10.0,
                    distance=100.0,
                    relative_bearing=10.0
                )
            },
            {
                'detection': Detection(
                    class_id=4,
                    class_name='traffic_camera',
                    confidence=0.85,
                    bbox=(500, 300, 600, 400)
                ),
                'position': MagicMock(
                    latitude=37.7750,
                    longitude=-122.4195,
                    altitude=15.0,
                    distance=150.0,
                    relative_bearing=-5.0
                )
            }
        ]
    
    def test_initialization(self):
        """Test mapper initialization."""
        self.assertEqual(self.mapper.distance_threshold, 10.0)
        self.assertEqual(len(self.mapper.mapped_objects), 0)
    
    def test_update(self):
        """Test updating the map with new objects."""
        # Add objects to the map
        mapped_objects = self.mapper.update(self.localized_objects, 1)
        
        # Check that objects were added
        self.assertEqual(len(mapped_objects), 2)
        
        # Check object properties
        obj_classes = sorted([obj.class_name for obj in mapped_objects])
        self.assertEqual(obj_classes, ['traffic_camera', 'traffic_light_red'])
        
        # Update with the same objects (should update existing)
        mapped_objects = self.mapper.update(self.localized_objects, 2)
        
        # Number of objects should still be 2
        self.assertEqual(len(mapped_objects), 2)
        
        # Check that frame numbers were updated
        for obj in mapped_objects:
            self.assertEqual(obj.last_seen, 2)
    
    def test_get_objects_by_class(self):
        """Test filtering objects by class."""
        # Add objects to the map
        self.mapper.update(self.localized_objects, 1)
        
        # Get traffic lights
        traffic_lights = self.mapper.get_objects_by_class('traffic_light')
        self.assertEqual(len(traffic_lights), 1)
        self.assertEqual(traffic_lights[0].class_name, 'traffic_light_red')
        
        # Get cameras
        cameras = self.mapper.get_objects_by_class('camera')
        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0].class_name, 'traffic_camera')
    
    def test_find_nearby_object(self):
        """Test finding nearby objects."""
        # Add objects to the map
        self.mapper.update(self.localized_objects, 1)
        
        # Find nearby traffic light
        nearby = self.mapper._find_nearby_object(
            37.7749001,  # Very close to first object
            -122.4194001,
            'traffic_light_red'
        )
        
        # Should find the object
        self.assertIsNotNone(nearby)
        self.assertEqual(nearby.class_name, 'traffic_light_red')
        
        # Find nearby object of wrong class
        nearby = self.mapper._find_nearby_object(
            37.7749001,
            -122.4194001,
            'traffic_camera'  # Wrong class
        )
        
        # Should not find anything
        self.assertIsNone(nearby)
        
        # Find nearby object that's too far
        nearby = self.mapper._find_nearby_object(
            37.7760,  # Too far
            -122.4205,
            'traffic_light_red'
        )
        
        # Should not find anything
        self.assertIsNone(nearby)
    
    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        # Test with known coordinates
        # San Francisco to Oakland ~ 12.24 km
        dist = self.mapper._haversine_distance(
            37.7749, -122.4194,  # San Francisco
            37.8044, -122.2711   # Oakland
        )
        
        # Check that distance is approximately correct
        self.assertAlmostEqual(dist / 1000, 12.24, delta=0.5)  # km
        
        # Test with same point (should be zero)
        dist = self.mapper._haversine_distance(
            37.7749, -122.4194,
            37.7749, -122.4194
        )
        
        self.assertAlmostEqual(dist, 0.0, places=8)


class TestMappedObject(unittest.TestCase):
    """Tests for the MappedObject class."""
    
    def setUp(self):
        """Set up test case."""
        # Create a mapped object
        self.obj = MappedObject(
            object_id='test_1',
            class_name='traffic_light_red',
            latitude=37.7749,
            longitude=-122.4194,
            state='red',
            confidence=0.95,
            altitude=10.0,
            first_seen=1,
            last_seen=1,
            count=1
        )
    
    def test_initialization(self):
        """Test object initialization."""
        self.assertEqual(self.obj.object_id, 'test_1')
        self.assertEqual(self.obj.class_name, 'traffic_light_red')
        self.assertEqual(self.obj.state, 'red')
        self.assertEqual(self.obj.count, 1)
        self.assertEqual(len(self.obj.position_history), 0)
    
    def test_update_position(self):
        """Test position updating."""
        # Initial position
        initial_lat = self.obj.latitude
        initial_lon = self.obj.longitude
        
        # Update position
        self.obj.update_position(37.7750, -122.4195, 15.0)
        
        # Check position history
        self.assertEqual(len(self.obj.position_history), 1)
        self.assertEqual(self.obj.position_history[0], (initial_lat, initial_lon))
        
        # Check new position
        self.assertEqual(self.obj.latitude, 37.7750)
        self.assertEqual(self.obj.longitude, -122.4195)
        self.assertEqual(self.obj.altitude, 15.0)
        
        # Add more positions and check median filtering
        for i in range(10):
            self.obj.update_position(37.7750 + 0.0001 * i, -122.4195 - 0.0001 * i)
        
        # Should have 10 positions in history (truncated to last 10)
        self.assertEqual(len(self.obj.position_history), 10)
        
        # Latest position should use median filtering
        self.assertNotEqual(self.obj.latitude, 37.7750 + 0.0001 * 9)  # Not the exact input
    
    def test_update_state(self):
        """Test state updating."""
        # Initial state
        self.assertEqual(self.obj.state, 'red')
        
        # Update state
        self.obj.update_state('green')
        
        # Check new state
        self.assertEqual(self.obj.state, 'green')
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        # Convert to dictionary
        obj_dict = self.obj.to_dict()
        
        # Check dictionary fields
        self.assertEqual(obj_dict['id'], 'test_1')
        self.assertEqual(obj_dict['class'], 'traffic_light_red')
        self.assertEqual(obj_dict['lat'], 37.7749)
        self.assertEqual(obj_dict['lon'], -122.4194)
        self.assertEqual(obj_dict['state'], 'red')
        self.assertEqual(obj_dict['confidence'], 0.95)
        self.assertEqual(obj_dict['first_seen'], 1)
        self.assertEqual(obj_dict['last_seen'], 1)
        self.assertEqual(obj_dict['count'], 1)


if __name__ == '__main__':
    unittest.main()